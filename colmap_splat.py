import os
import torch
import cv2
import numpy as np

from torchmetrics.image import StructuralSimilarityIndexMeasure

from dataloader import ColmapData
from options import GaussianSplattingOptions

from cuda_autograd_functions import (
    CameraPointProjection,
    ComputeSigmaWorld,
    ComputeGaussianProjectionJacobian,
    ComputeSigmaImage,
    RenderImage,
)
from utils import transform_points_torch
from structs import Gaussians, SimpleTimer, Tiles
from tile_culling import (
    match_gaussians_to_tiles_gpu,
    sort_gaussians,
)

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

base_lr = 0.002
optimizer = torch.optim.Adam(
    [
        {"params": gaussians.xyz, "lr": base_lr * 10},
        {"params": gaussians.quaternions, "lr": base_lr * 20},
        {"params": gaussians.scales, "lr": base_lr * 10},
        {"params": gaussians.opacities, "lr": base_lr * 1},
        {"params": gaussians.rgb, "lr": base_lr * 1},
    ],
)
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(torch.device("cuda"))

for i in range(501):
    print("Iteration: ", i)

    image_idx = 0
    world_T_image = images[image_idx].world_T_image

    optimizer.zero_grad()

    with SimpleTimer("Render Image (CUDA)"):
        xyz_camera_frame = transform_points_torch(gaussians.xyz, world_T_image)

        # perform frustrum culling
        culling_mask = torch.zeros(
            xyz_camera_frame.shape[0], dtype=torch.bool, device=gaussians.xyz.device
        )
        near_thresh = 0.3
        culling_mask = culling_mask | (xyz_camera_frame[:, 2] < near_thresh)
        uv = CameraPointProjection.apply(xyz_camera_frame, camera.K)

        culling_mask = (
            culling_mask
            | (uv[:, 0] < 0)
            | (uv[:, 0] > camera.width)
            | (uv[:, 1] < 0)
            | (uv[:, 1] > camera.height)
        )

        # cull gaussians outside of camera frustrum
        uv = uv[~culling_mask, :]
        xyz_camera_frame = xyz_camera_frame[~culling_mask, :]

        culled_gaussians = Gaussians(
            xyz=gaussians.xyz[~culling_mask, :],
            quaternions=gaussians.quaternions[~culling_mask, :],
            scales=gaussians.scales[~culling_mask, :],
            opacities=gaussians.opacities[~culling_mask],
            rgb=gaussians.rgb[~culling_mask, :],
        )

        sigma_world = ComputeSigmaWorld.apply(
            culled_gaussians.quaternions, culled_gaussians.scales
        )
        J = ComputeGaussianProjectionJacobian.apply(xyz_camera_frame, camera.K)
        sigma_image = ComputeSigmaImage.apply(sigma_world, J, world_T_image)

        # perform tile culling
        tiles = Tiles(camera.height, camera.width, uv.device)
        (
            gaussian_idx_by_splat_idx,
            splat_start_end_idx_by_tile_idx,
            tile_idx_by_splat_idx,
        ) = match_gaussians_to_tiles_gpu(uv, tiles, sigma_image)
        sorted_gaussian_indices = sort_gaussians(
            xyz_camera_frame, gaussian_idx_by_splat_idx, tile_idx_by_splat_idx
        )
        image = RenderImage.apply(
            culled_gaussians.rgb,
            culled_gaussians.opacities,
            uv,
            sigma_image,
            splat_start_end_idx_by_tile_idx,
            gaussian_idx_by_splat_idx,
            torch.tensor([camera.height, camera.width], device=uv.device),
        )

    gt_image = images[image_idx].image.to(torch.float32) / 255.0
    gt_image = gt_image.to(torch.device("cuda"))

    l2_loss = torch.nn.functional.mse_loss(image, gt_image)
    psnr = -10 * torch.log10(l2_loss)
    print("PSNR: ", psnr.detach().cpu().numpy())

    ssim_loss = 1.0 - ssim(
        image.unsqueeze(0).permute(0, 3, 1, 2),
        gt_image.unsqueeze(0).permute(0, 3, 1, 2),
    )

    loss = 0.8 * l2_loss + 0.2 * ssim_loss

    with SimpleTimer("Backward"):
        loss.backward()
    with SimpleTimer("Optimizer Step"):
        optimizer.step()

    if (i) % 10 == 0:
        with SimpleTimer("Save Images"):
            image = image.clip(0, 1).detach().cpu().numpy()
            cv2.imwrite(
                "colmap_splat/image_{}_iter{}.png".format(image_idx, i),
                (image * 255).astype(np.uint8)[..., ::-1],
            )
cv2.imwrite(
    "colmap_splat/image_{}_original.png".format(image_idx),
    (images[image_idx].image).detach().cpu().numpy()[..., ::-1],
)
