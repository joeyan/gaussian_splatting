import torch
import math

from utils import quaternion_to_rotation_torch, transform_points_torch
from structs import Tiles


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

    # the paper uses a plane at z=1 as the focal plane
    # so we need to premultiply by the focal length
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
    culling_mask = torch.zeros(
        gaussians.xyz.shape[0], dtype=torch.bool, device=gaussians.xyz.device
    )

    # transform gaussian centers to camera frame
    xyz_camera_frame = transform_points_torch(gaussians.xyz, world_T_image)

    culling_mask = culling_mask | (xyz_camera_frame[:, 2] < 1e-6)

    # project 3D gaussian centers to 2D
    homogenous_xy1 = torch.div(xyz_camera_frame, xyz_camera_frame[:, 2].unsqueeze(1))
    uv = torch.matmul(camera.K, homogenous_xy1.transpose(0, 1)).transpose(0, 1)[:, :2]

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
    for gaussian_idx in range(uvs.shape[0]):
        if global_culling_mask[gaussian_idx]:
            gaussian_to_tile_idx.append({})
            continue

        tile_x = torch.floor(uvs[gaussian_idx, 0] / tiles.tile_edge_size).int()
        tile_y = torch.floor(uvs[gaussian_idx, 1] / tiles.tile_edge_size).int()
        tile_index = tile_y * tiles.x_tiles_count + tile_x
        # add the tile that the gaussian projects into
        gaussian_to_tile_idx.append({tile_index.item()})

        a = sigma_image[gaussian_idx, 0, 0]
        b = sigma_image[gaussian_idx, 0, 1]
        c = sigma_image[gaussian_idx, 1, 0]
        d = sigma_image[gaussian_idx, 1, 1]

        # compute the two radii of the 2d gaussian
        left = (a + d) / 2
        right = torch.sqrt(torch.square(a - d) / 4 + b * c)
        r1 = torch.sqrt(left + right)
        r2 = torch.sqrt(left - right)
        # use the larger of the two radii, three sigma = 99.7%, for simplicity use a square, axis-aligned bounding box
        max_r = 3 * torch.max(r1, r2)

        bbox_top = uvs[gaussian_idx, 1] - max_r
        bbox_bottom = uvs[gaussian_idx, 1] + max_r
        bbox_left = uvs[gaussian_idx, 0] - max_r
        bbox_right = uvs[gaussian_idx, 0] + max_r

        # add the tiles that the bounding box intersects
        for tile_x in range(tiles.x_tiles_count):
            for tile_y in range(tiles.y_tiles_count):
                tile_index = tile_y * tiles.x_tiles_count + tile_x

                tile_top = tiles.tile_corners[tile_index, 0, 1]
                tile_bottom = tiles.tile_corners[tile_index, 1, 1]
                tile_left = tiles.tile_corners[tile_index, 0, 0]
                tile_right = tiles.tile_corners[tile_index, 1, 0]

                # there needs to be overlap in both x and y
                if (bbox_top < tile_bottom or bbox_bottom > tile_top) and (
                    bbox_left < tile_right or bbox_right > tile_left
                ):
                    gaussian_to_tile_idx[gaussian_idx].add(tile_index)

    return gaussian_to_tile_idx


def render_tiles(
    uvs,
    gaussians,
    sigma_image,
    culling_mask,
    camera,
):

    tiles = Tiles(camera.height, camera.width, uvs.device)

    gaussian_to_tile_idx = match_gaussians_to_tiles(
        uvs, culling_mask, tiles, sigma_image
    )
    image = torch.zeros(
        tiles.image_height_padded,
        tiles.image_width_padded,
        3,
        dtype=torch.float32,
        device=uvs.device,
    )
    alpha_accum = torch.zeros(
        tiles.image_height_padded,
        tiles.image_width_padded,
        dtype=torch.float32,
        device=uvs.device,
    )

    n_gaussians = uvs.shape[0]
    for idx in range(n_gaussians):
        if culling_mask[idx]:
            continue
        if len(gaussian_to_tile_idx[idx]) == 0:
            continue

        uv = uvs[idx, :]

        a = sigma_image[idx, 0, 0]
        b = sigma_image[idx, 0, 1]
        c = sigma_image[idx, 1, 0]
        d = sigma_image[idx, 1, 1]
        det = a * d - b * c

        opa = gaussians.opacities[idx]
        color = gaussians.rgb[idx]

        if det < 1e-14:
            print("det < 1e-14 at index: ", idx)
            continue

        tile_indexes = gaussian_to_tile_idx[idx]

        for tile_idx in tile_indexes:
            for row_offset in range(tiles.tile_edge_size):
                for col_offset in range(tiles.tile_edge_size):
                    row = tiles.tile_corners[tile_idx, 0, 1] + row_offset
                    col = tiles.tile_corners[tile_idx, 0, 0] + col_offset

                    if alpha_accum[row, col] > 0.99:
                        continue

                    uv_pixel = torch.tensor(
                        [col, row], dtype=torch.float32, device=uvs.device
                    )
                    uv_diff = uv_pixel - uv

                    mh_dist_sq = torch.dot(
                        uv_diff, torch.matmul(torch.inverse(sigma_image[idx]), uv_diff)
                    )

                    # Instead of computing the probability, just take the numerator.
                    # the numerator is a "normalized" normal distribution where the peak is 1
                    # this way the opacity and scale of the gaussian are decoupled
                    # additionally, this allows opacity based culling to behave similarly for
                    # gaussians with different scales
                    # prob = torch.exp(-0.5 * mh_dist_sq) / (2 * math.pi * torch.sqrt(det))
                    prob = torch.exp(-0.5 * mh_dist_sq)
                    if prob < 1e-24:
                        continue

                    alpha = opa * prob
                    weight = alpha * (1.0 - alpha_accum[row, col])

                    image[row, col, :] += color * weight
                    alpha_accum[row, col] += weight.squeeze(0)

    return image
