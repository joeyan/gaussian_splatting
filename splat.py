import torch

from tile_culling import (
    match_gaussians_to_tiles,
    match_gaussians_to_tiles_gpu,
    sort_gaussians,
)
from structs import Tiles, SimpleTimer
from splat_cuda import render_tiles_cuda


def render_tiles(
    uvs,
    gaussians,
    sigma_image,
    camera,
    xyz_camera_frame,
):

    tiles = Tiles(camera.height, camera.width, uvs.device)

    (
        gaussian_idx_by_splat_idx,
        splat_start_end_idx_by_tile_idx,
        tile_idx_by_splat_idx,
    ) = match_gaussians_to_tiles(uvs, tiles, sigma_image)

    sorted_gaussian_indices = sort_gaussians(
        xyz_camera_frame, gaussian_idx_by_splat_idx, tile_idx_by_splat_idx
    )

    image = torch.zeros(
        tiles.image_height,
        tiles.image_width,
        3,
        dtype=torch.float32,
        device=uvs.device,
    )

    with SimpleTimer("\tCPU render"):
        # iterate through each tile
        for tile_idx in range(tiles.tile_count):
            start_index = splat_start_end_idx_by_tile_idx[tile_idx]
            end_index = splat_start_end_idx_by_tile_idx[tile_idx + 1]

            # iterate through each pixel in the tile
            for row_offset in range(tiles.tile_edge_size):
                for col_offset in range(tiles.tile_edge_size):
                    row = tiles.tile_corners[tile_idx, 0, 1] + row_offset
                    col = tiles.tile_corners[tile_idx, 0, 0] + col_offset

                    alpha_accum = 0.0

                    # splat each gaussian for each pixel
                    for list_idx in range(start_index, end_index):
                        idx = sorted_gaussian_indices[list_idx]
                        if alpha_accum > 0.99:
                            continue

                        uv_pixel = torch.tensor(
                            [col, row], dtype=torch.float32, device=uvs.device
                        )
                        uv_diff = uv_pixel - uvs[idx, :]

                        mh_dist_sq = torch.dot(
                            uv_diff,
                            torch.matmul(torch.inverse(sigma_image[idx]), uv_diff),
                        )

                        # Instead of computing the probability, just take the numerator.
                        # the numerator is a "normalized" normal distribution where the peak is 1
                        # this way the opacity and scale of the gaussian are decoupled
                        # additionally, this allows opacity based culling to behave similarly for
                        # gaussians with different scales
                        prob = torch.exp(-0.5 * mh_dist_sq)
                        if prob < 1e-14:
                            continue

                        alpha = gaussians.opacities[idx] * prob
                        weight = alpha * (1.0 - alpha_accum)

                        image[row, col, :] += gaussians.rgb[idx] * weight
                        alpha_accum += weight.squeeze(0)
    return image


def render_tiles_gpu(
    uvs,
    gaussians,
    sigma_image,
    camera,
    xyz_camera_frame,
):
    with SimpleTimer("\tCreate Tiles"):
        tiles = Tiles(camera.height, camera.width, uvs.device)

    with SimpleTimer("\tGPU tile matching"):
        (
            gaussian_idx_by_splat_idx,
            splat_start_end_idx_by_tile_idx,
            tile_idx_by_splat_idx,
        ) = match_gaussians_to_tiles_gpu(uvs, tiles, sigma_image, mh_dist=3.0)

    with SimpleTimer("\tSorting Gaussians"):
        sorted_gaussian_indices = sort_gaussians(
            xyz_camera_frame, gaussian_idx_by_splat_idx, tile_idx_by_splat_idx
        )

    with SimpleTimer("\tCreate Image"):
        image = torch.zeros(
            tiles.image_height,
            tiles.image_width,
            3,
            dtype=torch.float32,
            device=uvs.device,
        )
        num_splats_per_pixel = torch.zeros(
            tiles.image_height,
            tiles.image_width,
            1,
            dtype=torch.int32,
            device=uvs.device,
        )
        final_weight_per_pixel = torch.zeros(
            tiles.image_height,
            tiles.image_width,
            1,
            dtype=torch.float32,
            device=uvs.device,
        )
    with SimpleTimer("\tGPU render"):
        render_tiles_cuda(
            uvs,
            gaussians.opacities,
            gaussians.rgb,
            sigma_image,
            splat_start_end_idx_by_tile_idx,
            sorted_gaussian_indices,
            num_splats_per_pixel,
            final_weight_per_pixel,
            image,
        )

    return image
