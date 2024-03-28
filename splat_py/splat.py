import torch

from splat_py.constants import *
from splat_py.utils import transform_points_torch, compute_rays_in_world_frame
from splat_py.cuda_autograd_functions import (
    CameraPointProjection,
    ComputeSigmaWorld,
    ComputeProjectionJacobian,
    ComputeSigmaImage,
    RenderImage,
)
from splat_py.structs import Gaussians, Tiles
from splat_py.tile_culling import (
    match_gaussians_to_tiles_gpu,
    sort_gaussians,
)


def splat(gaussians, world_T_image, camera):
    xyz_camera_frame = transform_points_torch(gaussians.xyz, world_T_image)
    uv = CameraPointProjection.apply(xyz_camera_frame, camera.K)

    # perform frustrum culling
    culling_mask = torch.zeros(
        xyz_camera_frame.shape[0],
        dtype=torch.bool,
        device=gaussians.xyz.device,
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
    uv = uv[~culling_mask, :]
    xyz_camera_frame = xyz_camera_frame[~culling_mask, :]

    culled_gaussians = Gaussians(
        xyz=gaussians.xyz[~culling_mask, :],
        quaternions=gaussians.quaternions[~culling_mask, :],
        scales=gaussians.scales[~culling_mask, :],
        opacities=torch.sigmoid(
            gaussians.opacities[~culling_mask]
        ),  # apply sigmoid activation to opacities
        rgb=gaussians.rgb[~culling_mask, :],
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
    ) = match_gaussians_to_tiles_gpu(uv, tiles, sigma_image, mh_dist=MH_DIST)

    sorted_gaussian_idx_by_splat_idx = sort_gaussians(
        xyz_camera_frame, gaussian_idx_by_splat_idx, tile_idx_by_splat_idx
    )
    rays = compute_rays_in_world_frame(camera, world_T_image)
    image = RenderImage.apply(
        culled_gaussians.rgb,
        culled_gaussians.opacities,
        uv,
        sigma_image,
        rays,
        splat_start_end_idx_by_tile_idx,
        sorted_gaussian_idx_by_splat_idx,
        torch.tensor([camera.height, camera.width], device=uv.device),
    )
    return image, culling_mask
