import torch
import cv2
import numpy as np

from torchmetrics.image import StructuralSimilarityIndexMeasure

from splat_py.constants import *
from splat_py.trainer import GSTrainer
from splat_py.dataloader import ColmapData
from splat_py.options import GaussianSplattingOptions
from splat_py.utils import transform_points_torch

from splat_py.cuda_autograd_functions import (
    CameraPointProjection,
    ComputeSigmaWorld,
    ComputeProjectionJacobian,
    ComputeSigmaImage,
    RenderImage,
)

from splat_py.structs import Gaussians, SimpleTimer, Tiles
from splat_py.tile_culling import (
    match_gaussians_to_tiles_gpu,
    sort_gaussians,
)

torch.manual_seed(0)
with SimpleTimer("Load Colmap Data"):
    colmap_data = ColmapData(
        "/home/joe/Downloads/garden", torch.device("cuda"), downsample_factor=4
    )
    options = GaussianSplattingOptions()
    gaussians = colmap_data.create_gaussians(options)

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

for i in range(7000):
    print("Iteration: ", i, "Num Gaussians: ", problem.gaussians.xyz.shape[0])

    # image_idx = 7
    image_idx = torch.randint(0, len(images), (1,)).item()
    world_T_image = images[image_idx].world_T_image

    problem.optimizer.zero_grad()

    with SimpleTimer("Render Image (CUDA)"):
        xyz_camera_frame = transform_points_torch(problem.gaussians.xyz, world_T_image)
        uv = CameraPointProjection.apply(xyz_camera_frame, camera.K)

        # perform frustrum culling
        culling_mask = torch.zeros(
            xyz_camera_frame.shape[0],
            dtype=torch.bool,
            device=problem.gaussians.xyz.device,
        )
        near_thresh = 0.3
        culling_mask = culling_mask | (xyz_camera_frame[:, 2] < near_thresh)
        culling_mask = (
            culling_mask
            | (uv[:, 0] < -1 * CULL_MASK_PADDING)
            | (uv[:, 0] > camera.width + CULL_MASK_PADDING)
            | (uv[:, 1] < -1 * CULL_MASK_PADDING)
            | (uv[:, 1] > camera.height + CULL_MASK_PADDING)
        )

        # cull gaussians outside of camera frustrum
        uv = uv[~culling_mask, :]
        xyz_camera_frame = xyz_camera_frame[~culling_mask, :]

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
        J = ComputeProjectionJacobian.apply(xyz_camera_frame, camera.K)
        sigma_image = ComputeSigmaImage.apply(sigma_world, J, world_T_image)

        # perform tile culling
        tiles = Tiles(camera.height, camera.width, uv.device)
        (
            gaussian_idx_by_splat_idx,
            splat_start_end_idx_by_tile_idx,
            tile_idx_by_splat_idx,
        ) = match_gaussians_to_tiles_gpu(uv, tiles, sigma_image, mh_dist=3.0)

        sorted_gaussian_idx_by_splat_idx = sort_gaussians(
            xyz_camera_frame, gaussian_idx_by_splat_idx, tile_idx_by_splat_idx
        )
        image = RenderImage.apply(
            culled_gaussians.rgb,
            culled_gaussians.opacities,
            uv,
            sigma_image,
            splat_start_end_idx_by_tile_idx,
            sorted_gaussian_idx_by_splat_idx,
            torch.tensor([camera.height, camera.width], device=uv.device),
        )

    gt_image = images[image_idx].image.to(torch.float32) / 255.0
    gt_image = gt_image.to(torch.device("cuda"))

    l1_loss = torch.nn.functional.l1_loss(image, gt_image)

    # print PSNR
    l2_loss = torch.nn.functional.mse_loss(image, gt_image)
    psnr = -10 * torch.log10(l2_loss)
    print("Iter: {}, PSNR: {}".format(i, psnr.detach().cpu().numpy()))

    # channel first tensor for SSIM
    ssim_loss = 1.0 - ssim(
        image.unsqueeze(0).permute(0, 3, 1, 2),
        gt_image.unsqueeze(0).permute(0, 3, 1, 2),
    )
    loss = 0.8 * l1_loss + 0.2 * ssim_loss
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

    if i > 150 and i % 100 == 0 and i < 5000:
        problem.adaptive_density_control(i)

    if (i) % 50 == 2 or i == 0:
        with SimpleTimer("Save Images"):
            image = image.clip(0, 1).detach().cpu().numpy()
            cv2.imwrite(
                "colmap_splat/iter{}_image_{}.png".format(i, image_idx),
                (image * 255).astype(np.uint8)[..., ::-1],
            )
