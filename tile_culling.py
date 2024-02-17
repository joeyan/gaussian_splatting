import torch
import math

from splat_cuda import compute_tiles_cuda, compute_tile_to_gaussian_vector
from structs import SimpleTimer


def compute_obb(
    uv,
    sigma_image,
    mh_dist=1.0,
):
    """
    https://cookierobotics.com/007/

    Compute the oriented bounding box of a 2D gaussian at a specific confidence level
    """
    a = sigma_image[0, 0]
    b = sigma_image[0, 1]
    c = sigma_image[1, 0]
    d = sigma_image[1, 1]

    # compute the two radii of the 2d gaussian
    left = (a + d) / 2
    right = torch.sqrt(torch.square(a - d) / 4 + b * c)
    lambda_1 = left + right
    r1 = mh_dist * torch.sqrt(lambda_1)  # major axis
    r2 = mh_dist * torch.sqrt(left - right)  # minor axis

    # compute angle of major axis
    # theta is ccw from +x axis
    if abs(b) < 1e-16:
        if a >= d:
            theta = 0
        else:
            theta = math.pi / 2
    else:
        theta = math.atan2((lambda_1 - a), b)

    R = torch.tensor(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=torch.float32,
        device=uv.device,
    )

    obb = torch.zeros(4, 2, dtype=torch.float32, device=uv.device)
    obb[0, :] = uv + R @ torch.tensor(
        [-r1, -r2], dtype=torch.float32, device=uv.device
    )  # top left aabb corner
    obb[1, :] = uv + R @ torch.tensor(
        [r1, -r2], dtype=torch.float32, device=uv.device
    )  # top right aabb corner
    obb[2, :] = uv + R @ torch.tensor(
        [-r1, r2], dtype=torch.float32, device=uv.device
    )  # bottom left aabb corner
    obb[3, :] = uv + R @ torch.tensor(
        [r1, r2], dtype=torch.float32, device=uv.device
    )  # bottom right aabb corner

    return obb


def compute_bbox_tile_intersection(
    bbox_gaussian,
    bbox_tile,
):
    """
    compute the intersection of a bbox and a tile bbox

    Use the split axis theorem with simplifications:
    1) There are 4 axes to check for overlap since there are two sets of parallel edges in each rectangle
    2) Taking the normal of each edge is not required since the edges in a rectangle are always perpendicular
    3) Two axes to check are the x and y axes since the tile bbox is axis aligned

    bbox format:
    torch.tensor([
        top left,
        top right,
        bottom left,
        bottom right
    ])

    For more details: https://dyn4j.org/2010/01/sat/
    """

    # check x axis for overlap
    min_x_gaussian = torch.min(bbox_gaussian[:, 0])
    max_x_gaussian = torch.max(bbox_gaussian[:, 0])
    min_x_tile = bbox_tile[0, 0]
    max_x_tile = bbox_tile[1, 0]

    if min_x_gaussian > max_x_tile or max_x_gaussian < min_x_tile:
        return False

    # check y axis
    min_y_gaussian = torch.min(bbox_gaussian[:, 1])
    max_y_gaussian = torch.max(bbox_gaussian[:, 1])
    min_y_tile = bbox_tile[0, 1]
    max_y_tile = bbox_tile[2, 1]

    if min_y_gaussian > max_y_tile or max_y_gaussian < min_y_tile:
        return False

    # bbox_gaussian axis 0: top left to top right
    axis_0 = bbox_gaussian[1, :] - bbox_gaussian[0, :]
    # bbox_gaussian axis 1: top left to bottom left
    axis_1 = bbox_gaussian[2, :] - bbox_gaussian[0, :]

    for axis in [axis_0, axis_1]:
        projected_gaussian_0 = torch.dot(axis, bbox_gaussian[0, :])
        projected_tile_0 = torch.dot(axis, bbox_tile[0, :])

        min_projected_gaussian = projected_gaussian_0
        max_projected_gaussian = projected_gaussian_0
        min_projected_tile = projected_tile_0
        max_projected_tile = projected_tile_0

        for i in range(1, 4):
            projected_gaussian_pt = torch.dot(axis, bbox_gaussian[i, :])
            min_projected_gaussian = torch.min(
                min_projected_gaussian, projected_gaussian_pt
            )
            max_projected_gaussian = torch.max(
                max_projected_gaussian, projected_gaussian_pt
            )

            projected_tile_pt = torch.dot(axis, bbox_tile[i, :])
            min_projected_tile = torch.min(min_projected_tile, projected_tile_pt)
            max_projected_tile = torch.max(max_projected_tile, projected_tile_pt)

        if (
            min_projected_gaussian > max_projected_tile
            or max_projected_gaussian < min_projected_tile
        ):
            return False

    return True


def match_gaussians_to_tiles(
    uvs,
    tiles,
    sigma_image,
):
    """
    Determine which tiles each gaussian is in
    """
    gaussian_to_tile_idx = []

    num_gaussians = torch.tensor(0, dtype=torch.int32, device=uvs.device)
    for gaussian_idx in range(uvs.shape[0]):
        gaussian_to_tile_idx.append([])

        bbox = compute_obb(
            uvs[gaussian_idx, :], sigma_image[gaussian_idx, :, :], mh_dist=3.0
        )
        # add the tiles that the bounding box intersects
        for tile_index in range(tiles.tile_count):
            if compute_bbox_tile_intersection(
                bbox, tiles.tile_corners[tile_index, :, :].float()
            ):
                gaussian_to_tile_idx[gaussian_idx].append(tile_index)
                num_gaussians += 1

    gaussian_indices_by_tile = torch.zeros(
        num_gaussians, dtype=torch.int32, device=uvs.device
    )
    gaussian_start_end_indices = torch.zeros(
        tiles.tile_count + 1, dtype=torch.int32, device=uvs.device
    )

    gaussian_start_end_indices[0] = 0
    current_index = 0

    for tile_idx in range(tiles.tile_count):
        gaussian_start_end_indices[tile_idx] = current_index
        for gaussian_idx in range(uvs.shape[0]):
            if tile_idx in gaussian_to_tile_idx[gaussian_idx]:
                gaussian_indices_by_tile[current_index] = gaussian_idx
                current_index += 1
    gaussian_start_end_indices[tiles.tile_count] = current_index

    return gaussian_indices_by_tile, gaussian_start_end_indices


def match_gaussians_to_tiles_gpu(
    uvs,
    tiles,
    sigma_image,
    mh_dist=3.0,
):
    max_gaussians = 4096
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
        gaussian_start_end_indices = torch.zeros(
            tiles.tile_count + 1, dtype=torch.int32, device=uvs.device
        )
        gaussian_start_end_indices[1:] = torch.cumsum(
            num_gaussians_per_tile.squeeze(), dim=0
        )

    # create gaussian to tile vector
    gaussian_indices_by_tile = torch.zeros(
        gaussian_start_end_indices[-1] + 1, dtype=torch.int32, device=uvs.device
    )

    with SimpleTimer("\t\tGPU Tile Vectorization"):
        compute_tile_to_gaussian_vector(
            gaussian_indices_per_tile,
            num_gaussians_per_tile,
            gaussian_start_end_indices,
            gaussian_indices_by_tile,
        )
    return gaussian_indices_by_tile, gaussian_start_end_indices
