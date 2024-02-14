import torch
import math

from utils import quaternion_to_rotation_torch, transform_points_torch
from structs import Tiles
import splat_cuda


def compute_sigma_world(
    gaussians,
):
    """
    Compute the 3x3 covariance matrix in world frame
    """
    # normalize quaternions and convert to rotation matrix
    normalized_q = gaussians.quaternions / torch.norm(
        gaussians.quaternions, dim=1, keepdim=True
    )
    R = quaternion_to_rotation_torch(normalized_q)  # N x 3 x 3

    # convert scale to 3x3 matrix and apply exponential activation
    # according to paper, this provides smoother gradients
    S = torch.diag_embed(torch.exp(gaussians.scales))

    RS = torch.bmm(R, S)  # N x 3 x 3
    sigma_world = torch.bmm(RS, RS.transpose(1, 2))  # N x 3 x 3
    return sigma_world


def compute_projection_jacobian(
    xyz_camera_frame,  # N x 3
    camera,
):
    """
    Compute the jacobian of the projection of a 3D gaussian

    [fx/z 0   -fx*x/(z*z)]
    [0   fy/z -fy*y/(z*z)]

    See the following references:
    1) The section right after equation (5) in the 3D Gaussian Splatting paper
    2) Equation (25) in the EWA Volume Splatting paper:
    https://www.cs.umd.edu/~zwicker/publications/EWAVolumeSplatting-VIS01.pdf
    """

    # the paper uses a plane at z=1 as the focal plane - so we need to scale by fx, fy
    # the third row is dropped because only the first two rows/cols are used
    # in the 2d covariance matrix

    fx = camera.K[0, 0]
    fy = camera.K[1, 1]

    for i in range(xyz_camera_frame.shape[0]):
        J = torch.tensor(
            [
                fx / xyz_camera_frame[i, 2],
                0,
                -1 * fx * xyz_camera_frame[i, 0] / torch.square(xyz_camera_frame[i, 2]),
                0,
                fy / xyz_camera_frame[i, 2],
                -1 * fy * xyz_camera_frame[i, 1] / torch.square(xyz_camera_frame[i, 2]),
            ],
            dtype=xyz_camera_frame.dtype,
            device=xyz_camera_frame.device,
        )
        J = J.reshape(2, 3)
        if i == 0:
            J_stack = J.unsqueeze(0)
        else:
            J_stack = torch.cat((J_stack, J.unsqueeze(0)), dim=0)
    return J_stack


def camera_projection(
    world_T_image,
    camera,
    gaussians,
):
    """
    Project 3D gaussians into 2D
    Cull gaussians outside of camera frustrum
    """
    n_gaussians = gaussians.xyz.shape[0]

    culling_mask = torch.zeros(
        n_gaussians, dtype=torch.bool, device=gaussians.xyz.device
    )

    # transform gaussian centers to camera frame
    xyz_camera_frame = transform_points_torch(gaussians.xyz, world_T_image)

    culling_mask = culling_mask | (xyz_camera_frame[:, 2] < 1e-6)

    uv = torch.zeros(
        n_gaussians, 2, dtype=torch.float32, device=xyz_camera_frame.device
    )
    for i in range(n_gaussians):
        uv[i, 0] = (
            camera.K[0, 0] * xyz_camera_frame[i, 0] / xyz_camera_frame[i, 2]
            + camera.K[0, 2]
        )
        uv[i, 1] = (
            camera.K[1, 1] * xyz_camera_frame[i, 1] / xyz_camera_frame[i, 2]
            + camera.K[1, 2]
        )

    culling_mask = (
        culling_mask
        | (uv[:, 0] < 0)
        | (uv[:, 0] > camera.width)
        | (uv[:, 1] < 0)
        | (uv[:, 1] > camera.height)
    )

    # compute sigma in world frame
    sigma_world = compute_sigma_world(gaussians)

    # projection jacobian
    J = compute_projection_jacobian(xyz_camera_frame, camera)

    # compute 2D gaussian: sigma_projected
    W = world_T_image[:3, :3].repeat(gaussians.xyz.shape[0], 1, 1)
    JW = torch.bmm(J, W)  # N x 2 x 3
    W_t_J_t = torch.bmm(W.transpose(1, 2), J.transpose(1, 2))  # N x 3 x 2
    JWSigma = torch.bmm(JW, sigma_world)  # N x 2 x 3
    JWSigmaW_tJ_t = torch.bmm(JWSigma, W_t_J_t)  # N x 2 x 2

    return uv, JWSigmaW_tJ_t, culling_mask


def compute_probability(
    uv_diff,
    sigma_image,
):
    """
    Compute the probability of a pixel given a 2D gaussian
    """
    mh_dist_sq = torch.dot(uv_diff, torch.matmul(torch.inverse(sigma_image), uv_diff))
    prob = torch.exp(-0.5 * mh_dist_sq) / (
        2 * math.pi * torch.sqrt(torch.det(sigma_image) + 1e-7)
    )
    return prob


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
    if b < 1e-4:
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
    global_culling_mask,
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
        if global_culling_mask[gaussian_idx]:
            continue

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


def render_tiles(
    uvs,
    gaussians,
    sigma_image,
    culling_mask,
    camera,
):

    tiles = Tiles(camera.height, camera.width, uvs.device)

    gaussian_indices_by_tile, gaussian_start_end_indices = match_gaussians_to_tiles(
        uvs, culling_mask, tiles, sigma_image
    )
    image = torch.zeros(
        tiles.image_height,
        tiles.image_width,
        3,
        dtype=torch.float32,
        device=uvs.device,
    )

    # iterate through each tile
    for tile_idx in range(tiles.tile_count):
        start_index = gaussian_start_end_indices[tile_idx]
        end_index = gaussian_start_end_indices[tile_idx + 1]

        # iterate through each pixel in the tile
        for row_offset in range(tiles.tile_edge_size):
            for col_offset in range(tiles.tile_edge_size):
                row = tiles.tile_corners[tile_idx, 0, 1] + row_offset
                col = tiles.tile_corners[tile_idx, 0, 0] + col_offset

                alpha_accum = 0.0

                # splat each gaussian for each pixel
                for list_idx in range(start_index, end_index):
                    idx = gaussian_indices_by_tile[list_idx]
                    if alpha_accum > 0.99:
                        continue

                    uv_pixel = torch.tensor(
                        [col, row], dtype=torch.float32, device=uvs.device
                    )
                    uv_diff = uv_pixel - uvs[idx, :]

                    mh_dist_sq = torch.dot(
                        uv_diff, torch.matmul(torch.inverse(sigma_image[idx]), uv_diff)
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
    culling_mask,
    camera,
):
    tiles = Tiles(camera.height, camera.width, uvs.device)

    gaussian_indices_by_tile, gaussian_start_end_indices = match_gaussians_to_tiles(
        uvs, culling_mask, tiles, sigma_image
    )
    image = torch.zeros(
        tiles.image_height,
        tiles.image_width,
        3,
        dtype=torch.float32,
        device=uvs.device,
    )
    splat_cuda.render_tiles_cuda(
        uvs,
        gaussians.opacities,
        gaussians.rgb,
        sigma_image,
        gaussian_start_end_indices,
        gaussian_indices_by_tile,
        image,
    )
    return image
