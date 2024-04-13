import torch

from splat_py.constants import *
from splat_py.utils import transform_points_torch, compute_rays_in_world_frame
from splat_py.cuda_autograd_functions import (
    CameraPointProjection,
    ComputeSigmaWorld,
    ComputeProjectionJacobian,
    ComputeConic,
    RenderImage,
    PrecomputeRGBFromSH,
)
from splat_py.structs import Gaussians, Tiles
from splat_py.tile_culling import (
    match_gaussians_to_tiles_gpu,
    sort_gaussians,
)


def rasterize(gaussians, world_T_image, camera):
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

    if gaussians.sh is not None:
        culled_gaussians = Gaussians(
            xyz=gaussians.xyz[~culling_mask, :],
            quaternion=gaussians.quaternion[~culling_mask, :],
            scale=gaussians.scale[~culling_mask, :],
            opacity=torch.sigmoid(
                gaussians.opacity[~culling_mask]
            ),  # apply sigmoid activation to opacity
            rgb=gaussians.rgb[~culling_mask, :],
            sh=gaussians.sh[~culling_mask, :],
        )
    else:
        culled_gaussians = Gaussians(
            xyz=gaussians.xyz[~culling_mask, :],
            quaternion=gaussians.quaternion[~culling_mask, :],
            scale=gaussians.scale[~culling_mask, :],
            opacity=torch.sigmoid(
                gaussians.opacity[~culling_mask]
            ),  # apply sigmoid activation to opacity
            rgb=gaussians.rgb[~culling_mask, :],
        )

    sigma_world = ComputeSigmaWorld.apply(culled_gaussians.quaternion, culled_gaussians.scale)
    J = ComputeProjectionJacobian.apply(xyz_camera_frame, camera.K)
    conic = ComputeConic.apply(sigma_world, J, world_T_image)

    # perform tile culling
    tiles = Tiles(camera.height, camera.width, uv.device)
    (
        gaussian_idx_by_splat_idx,
        splat_start_end_idx_by_tile_idx,
        tile_idx_by_splat_idx,
    ) = match_gaussians_to_tiles_gpu(uv, tiles, conic, mh_dist=MH_DIST)

    sorted_gaussian_idx_by_splat_idx = sort_gaussians(
        xyz_camera_frame, gaussian_idx_by_splat_idx, tile_idx_by_splat_idx
    )
    rays = torch.zeros(1, 1, 1, dtype=gaussians.xyz.dtype, device=gaussians.xyz.device)
    if culled_gaussians.sh is not None:
        sh_coeffs = torch.cat((culled_gaussians.rgb.unsqueeze(dim=2), culled_gaussians.sh), dim=2)
        if USE_SH_PRECOMPUTE:
            render_rgb = PrecomputeRGBFromSH.apply(
                sh_coeffs, culled_gaussians.xyz, torch.inverse(world_T_image).contiguous()
            )
        else:
            render_rgb = sh_coeffs
            # actually need to compute rays here
            rays = compute_rays_in_world_frame(camera, world_T_image)
    else:
        render_rgb = culled_gaussians.rgb

    image = RenderImage.apply(
        render_rgb,
        culled_gaussians.opacity,
        uv,
        conic,
        rays,
        splat_start_end_idx_by_tile_idx,
        sorted_gaussian_idx_by_splat_idx,
        torch.tensor([camera.height, camera.width], device=uv.device),
    )
    return image, culling_mask, uv
