import torch

from splat_cuda import (
    compute_tiles_cuda,
    compute_splat_to_gaussian_id_vector_cuda,
)
from splat_py.structs import SimpleTimer


def match_gaussians_to_tiles_gpu(
    uvs,
    tiles,
    sigma_image,
    mh_dist,
):
    max_gaussians = max(uvs.shape[0] // 10, 1024)
    gaussian_indices_per_tile = (
        torch.ones(
            tiles.tile_count, max_gaussians, dtype=torch.int32, device=uvs.device
        )
        * -1
    )
    num_gaussians_per_tile = torch.zeros(
        tiles.tile_count, 1, dtype=torch.int32, device=uvs.device
    )

    with SimpleTimer("\t\tGPU compute tiles"):
        compute_tiles_cuda(
            uvs,
            sigma_image,
            tiles.x_tiles_count,
            tiles.y_tiles_count,
            mh_dist,
            gaussian_indices_per_tile,
            num_gaussians_per_tile,
        )

    with SimpleTimer("\t\tCreate outputs for GPU Tile Vectorization"):
        # create start/end indices
        splat_start_end_idx_by_tile_idx = torch.zeros(
            tiles.tile_count + 1, dtype=torch.int32, device=uvs.device
        )
        splat_start_end_idx_by_tile_idx[1:] = torch.cumsum(
            num_gaussians_per_tile.squeeze(), dim=0
        )

    num_splats = splat_start_end_idx_by_tile_idx[-1]
    # create gaussian to tile vector
    gaussian_idx_by_splat_idx = (
        torch.ones(num_splats, dtype=torch.int32, device=uvs.device) * -1
    )
    tile_idx_by_splat_idx = (
        torch.ones(num_splats, dtype=torch.int32, device=uvs.device) * -1
    )

    with SimpleTimer("\t\tGPU Tile Vectorization"):
        compute_splat_to_gaussian_id_vector_cuda(
            gaussian_indices_per_tile,
            num_gaussians_per_tile,
            splat_start_end_idx_by_tile_idx,
            gaussian_idx_by_splat_idx,
            tile_idx_by_splat_idx,
        )
    return (
        gaussian_idx_by_splat_idx,
        splat_start_end_idx_by_tile_idx,
        tile_idx_by_splat_idx,
    )


def sort_gaussians(
    xyz_camera_frame,
    gaussian_idx_by_splat_idx,
    tile_idx_by_splat_idx,
):
    if not xyz_camera_frame.is_contiguous():
        xyz_camera_frame = xyz_camera_frame.contiguous()
    if not gaussian_idx_by_splat_idx.is_contiguous():
        gaussian_idx_by_splat_idx = gaussian_idx_by_splat_idx.contiguous()
    if not tile_idx_by_splat_idx.is_contiguous():
        tile_idx_by_splat_idx = tile_idx_by_splat_idx.contiguous()

    # sort gaussians within each tile for front to back rendering
    max_depth = torch.max(xyz_camera_frame[:, 2])
    if torch.any(torch.isnan(max_depth)):
        print("max_depth is NaN")
        exit()
    depth_per_splat = (xyz_camera_frame[gaussian_idx_by_splat_idx])[:, 2]
    # key should be tile_id and depth in camera frame (z) so the gaussians are still associated with the correct tile
    sort_keys = depth_per_splat.to(torch.float32) + (
        max_depth + 1.0
    ) * tile_idx_by_splat_idx.to(torch.float32)

    if not sort_keys.is_contiguous():
        sort_keys = sort_keys.contiguous()

    _, sorted_indices = torch.sort(sort_keys, descending=False)
    sorted_gaussian_indices = gaussian_idx_by_splat_idx[sorted_indices]

    if not sorted_gaussian_indices.is_contiguous():
        sorted_gaussian_indices = sorted_gaussian_indices.contiguous()

    return sorted_gaussian_indices
