import os
import numpy as np
import cv2
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure

from splat_py.constants import *
from splat_py.dataloader import ColmapData
from splat_py.utils import transform_points_torch
from splat_py.cuda_autograd_functions import (
    CameraPointProjection,
    ComputeSigmaWorld,
    ComputeProjectionJacobian,
    ComputeSigmaImage,
    RenderImage,
)
from splat_py.structs import Gaussians, Tiles, SimpleTimer
from splat_py.tile_culling import (
    match_gaussians_to_tiles_gpu,
    sort_gaussians,
)
from splat_py.trainer import GSTrainer
from splat_py.splat import splat

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

torch.manual_seed(0)
with SimpleTimer("Load Colmap Data"):
    colmap_data = ColmapData(DATASET_PATH, torch.device("cuda"), downsample_factor=4)
    gaussians = colmap_data.create_gaussians()

    images = colmap_data.get_images()
    cameras = colmap_data.get_cameras()
    camera = cameras[1]

    if SAVE_ADAPTIVE_CONTROL_DEBUG:
        torch.save(camera.K, "{}/camera_K.pth".format(OUTPUT_DIR))
        for i in range(len(images)):
            torch.save(
                images[i].world_T_image, "{}/world_T_image_{}.pth".format(OUTPUT_DIR, i)
            )

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
        xyz_camera_frame = transform_points_torch(problem.gaussians.xyz, world_T_image)
        uv = CameraPointProjection.apply(xyz_camera_frame, camera.K)
        uv.retain_grad()

        # perform frustrum culling
        culling_mask = torch.zeros(
            xyz_camera_frame.shape[0],
            dtype=torch.bool,
            device=problem.gaussians.xyz.device,
        )
        culling_mask = culling_mask | (xyz_camera_frame[:, 2] < NEAR_THRESH)
        culling_mask = (
            culling_mask
            | (uv[:, 0] < -1 * CULL_MASK_PADDING)
            | (uv[:, 0] > camera.width + CULL_MASK_PADDING)
            | (uv[:, 1] < -1 * CULL_MASK_PADDING)
            | (uv[:, 1] > camera.height + CULL_MASK_PADDING)
        )

        # cull gaussians outside of camera frustrum
        culled_uv = uv[~culling_mask, :]
        culled_xyz_camera_frame = xyz_camera_frame[~culling_mask, :]

        culled_gaussians = Gaussians(
            xyz=problem.gaussians.xyz[~culling_mask, :],
            quaternions=problem.gaussians.quaternions[~culling_mask, :],
            scales=problem.gaussians.scales[~culling_mask, :],
            opacities=torch.sigmoid(
                problem.gaussians.opacities[~culling_mask]
            ),  # apply sigmoid activation to opacities
            rgb=torch.sigmoid(
                problem.gaussians.rgb[~culling_mask, :]
            ),  # apply sigmoid activation to rgb
        )
        sigma_world = ComputeSigmaWorld.apply(
            culled_gaussians.quaternions, culled_gaussians.scales
        )
        J = ComputeProjectionJacobian.apply(culled_xyz_camera_frame, camera.K)
        sigma_image = ComputeSigmaImage.apply(sigma_world, J, world_T_image)

        # perform tile culling
        tiles = Tiles(camera.height, camera.width, culled_uv.device)
        (
            gaussian_idx_by_splat_idx,
            splat_start_end_idx_by_tile_idx,
            tile_idx_by_splat_idx,
        ) = match_gaussians_to_tiles_gpu(culled_uv, tiles, sigma_image, mh_dist=MH_DIST)

        sorted_gaussian_idx_by_splat_idx = sort_gaussians(
            culled_xyz_camera_frame, gaussian_idx_by_splat_idx, tile_idx_by_splat_idx
        )
        image = RenderImage.apply(
            culled_gaussians.rgb,
            culled_gaussians.opacities,
            culled_uv,
            sigma_image,
            splat_start_end_idx_by_tile_idx,
            sorted_gaussian_idx_by_splat_idx,
            torch.tensor([camera.height, camera.width], device=culled_uv.device),
        )

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

    problem.uv_grad_accum += torch.abs(uv.grad.detach())
    problem.xyz_grad_accum += torch.abs(problem.gaussians.xyz.grad.detach())
    problem.grad_accum_count += (~culling_mask).int()

    if torch.any(~torch.isfinite(problem.uv_grad_accum)):
        print(
            "uv_grad_accum NaN or Inf {}".format(
                torch.sum(~torch.isfinite(problem.uv_grad_accum)).detach()
            )
        )

    if torch.any(~torch.isfinite(problem.xyz_grad_accum)):
        print(
            "xyz_grad_accum NaN or Inf {}".format(
                torch.sum(~torch.isfinite(problem.xyz_grad_accum)).detach()
            )
        )

    # make sure to perform test split eval before adaptive control
    # if adaptive control occurs in the same iteration, test psnr will be low
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
        if SAVE_ADAPTIVE_CONTROL_DEBUG and i > 1500:
            torch.save(
                {
                    "uv_grad_accum": problem.uv_grad_accum,
                    "xyz_grad_accum": problem.xyz_grad_accum,
                    "grad_accum_count": problem.grad_accum_count,
                },
                "{}/iter{}_adaptive_control_debug_data.pth".format(OUTPUT_DIR, i),
            )
            torch.save(
                problem.gaussians,
                "{}/iter{}_adaptive_control_gaussians.pth".format(OUTPUT_DIR, i),
            )
            exit()
        problem.adaptive_density_control_update_adam(i)

    if (
        i > RESET_OPACTIY_START
        and i < RESET_OPACITY_END
        and i % RESET_OPACITY_INTERVAL == 0
    ):
        problem.reset_opacities(i)

    if i % SAVE_DEBUG_IMAGE_INTERVAL == 0:
        with SimpleTimer("Save Images"):
            debug_image = image.clip(0, 1).detach().cpu().numpy()
            cv2.imwrite(
                "{}/iter{}_image_{}.png".format(OUTPUT_DIR, i, image_idx),
                (debug_image * SATURATED_PIXEL_VALUE).astype(np.uint8)[..., ::-1],
            )
