import torch
import cv2
import numpy as np

from torchmetrics.image import StructuralSimilarityIndexMeasure

from splat_py.constants import *
from splat_py.trainer import GSTrainer
from splat_py.dataloader import ColmapData
from splat_py.structs import SimpleTimer
from splat_py.splat import splat

torch.manual_seed(0)
with SimpleTimer("Load Colmap Data"):
    colmap_data = ColmapData(
        "/home/joe/Downloads/garden", torch.device("cuda"), downsample_factor=4
    )
    gaussians = colmap_data.create_gaussians()

    images = colmap_data.get_images()
    cameras = colmap_data.get_cameras()
    camera = cameras[1]

    gaussians.xyz = torch.nn.Parameter(gaussians.xyz)
    gaussians.quaternions = torch.nn.Parameter(gaussians.quaternions)
    gaussians.scales = torch.nn.Parameter(gaussians.scales)
    gaussians.opacities = torch.nn.Parameter(gaussians.opacities)
    gaussians.rgb = torch.nn.Parameter(gaussians.rgb)

problem = GSTrainer(gaussians)
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(torch.device("cuda"))

num_images = len(images)
image_list = np.arange(num_images)
test_split = np.arange(0, num_images, TEST_SPLIT_RATIO)
train_split = np.array(list(set(image_list) - set(test_split)))
num_train_images = train_split

for i in range(NUM_ITERS):
    image_idx = np.random.choice(train_split)
    world_T_image = images[image_idx].world_T_image

    problem.optimizer.zero_grad()
    with SimpleTimer("Render Image (CUDA)"):
        image, culling_mask = splat(problem.gaussians, world_T_image, camera)

    gt_image = images[image_idx].image.to(torch.float32) / SATURATED_PIXEL_VALUE
    gt_image = gt_image.to(torch.device("cuda"))

    l1_loss = torch.nn.functional.l1_loss(image, gt_image)

    # print PSNR
    l2_loss = torch.nn.functional.mse_loss(image, gt_image)
    psnr = -10 * torch.log10(l2_loss)

    if i % PRINT_INTERVAL == 0:
        print(
            "Iter: {}, PSNR: {}, N: {}".format(
                i, psnr.detach().cpu().numpy(), problem.gaussians.xyz.shape[0]
            )
        )

    # channel first tensor for SSIM
    ssim_loss = 1.0 - ssim(
        image.unsqueeze(0).permute(0, 3, 1, 2),
        gt_image.unsqueeze(0).permute(0, 3, 1, 2),
    )
    loss = (1.0 - SSIM_RATIO) * l1_loss + SSIM_RATIO * ssim_loss
    with SimpleTimer("Backward"):
        loss.backward()
    with SimpleTimer("Optimizer Step"):
        problem.optimizer.step()

    problem.pos_grad_accum += torch.abs(problem.gaussians.xyz.grad)
    problem.grad_accum_count += (~culling_mask).int()

    if torch.any(~torch.isfinite(problem.pos_grad_accum)):
        print(
            "pos_grad_accum NaN or Inf {}".format(
                torch.sum(~torch.isfinite(problem.pos_grad_accum)).detach()
            )
        )

    # make sure to perform test split eval before adaptive control in case they are done at the same iter
    if i % TEST_EVAL_INTERVAL == 0:
        with torch.no_grad():
            test_psnrs = []
            for test_img_idx in test_split:
                test_world_T_image = images[test_img_idx].world_T_image
                test_image, _ = splat(problem.gaussians, world_T_image, camera)
                gt_image = (
                    images[image_idx].image.to(torch.float32) / SATURATED_PIXEL_VALUE
                )
                gt_image = gt_image.to(torch.device("cuda"))
                l2_loss = torch.nn.functional.mse_loss(test_image, gt_image)
                psnr = -10 * torch.log10(l2_loss).item()
                test_psnrs.append(psnr)
            print("\tTEST SPLIT PSNR: ", np.mean(np.array(test_psnrs)))

    if (
        i > ADAPTIVE_CONTROL_START
        and i % ADAPTIVE_CONTROL_INVERVAL == 0
        and i < ADAPTIVE_CONTROL_END
    ):
        problem.adaptive_density_control(i)

    if i % SAVE_DEBUG_IMAGE_INTERVAL == 0:
        with SimpleTimer("Save Images"):
            image = image.clip(0, 1).detach().cpu().numpy()
            cv2.imwrite(
                "colmap_splat/iter{}_image_{}.png".format(i, image_idx),
                (image * SATURATED_PIXEL_VALUE).astype(np.uint8)[..., ::-1],
            )
