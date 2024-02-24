import torch
from utils import quaternion_to_rotation_torch, transform_points_torch

from splat_cuda import (
    camera_projection_cuda,
    compute_sigma_world_cuda,
    compute_projection_jacobian_cuda,
    compute_sigma_image_cuda,
)
from structs import Gaussians, SimpleTimer


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


def project_points(
    world_T_image,
    camera,
    gaussians,
    near_thresh=0.3,
):
    n_gaussians = gaussians.xyz.shape[0]

    culling_mask = torch.zeros(
        n_gaussians, dtype=torch.bool, device=gaussians.xyz.device
    )

    # transform gaussian centers to camera frame
    with SimpleTimer("\tTransform to Camera Frame"):
        xyz_camera_frame = transform_points_torch(gaussians.xyz, world_T_image)

    culling_mask = culling_mask | (xyz_camera_frame[:, 2] < near_thresh)

    uv = torch.zeros(
        n_gaussians, 2, dtype=torch.float32, device=xyz_camera_frame.device
    )

    if gaussians.xyz.device != torch.device("cpu"):
        with SimpleTimer("\tGPU projection"):
            camera_projection_cuda(xyz_camera_frame[:, :3], camera.K, uv)
    else:
        with SimpleTimer("\tCPU projection"):
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
    return uv, xyz_camera_frame, culling_mask


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

    the paper uses a plane at z=1 as the focal plane - so we need to scale by fx, fy
    the third row is dropped because only the first two rows/cols are used
    in the 2d covariance matrix

    """
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


def compute_sigma_image(
    gaussians,
    xyz_camera_frame,
    camera,
    world_T_image,
):
    """
    Compute the 2D covariance matrix in image frame
    """
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
    return JWSigmaW_tJ_t


def project_and_cull(
    world_T_image,
    camera,
    gaussians,
):
    """
    Project 3D gaussians into 2D
    Cull gaussians outside of camera frustrum
    """
    uv, xyz_camera_frame, culling_mask = project_points(
        world_T_image, camera, gaussians
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

    with SimpleTimer("\tProject Covariance Matrix"):
        sigma_image = compute_sigma_image(
            culled_gaussians, xyz_camera_frame, camera, world_T_image
        )

    return uv, sigma_image, culled_gaussians, xyz_camera_frame


def project_and_cull_cuda(
    world_T_image,
    camera,
    gaussians,
):
    uv, xyz_camera_frame, culling_mask = project_points(
        world_T_image, camera, gaussians
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

    n_culled_gaussians = culled_gaussians.xyz.shape[0]
    device = culled_gaussians.xyz.device

    with SimpleTimer("\tProject Covariance Matrix GPU"):
        sigma_world = torch.zeros(
            (n_culled_gaussians, 3, 3), dtype=torch.float32, device=device
        )
        compute_sigma_world_cuda(
            culled_gaussians.quaternions,
            culled_gaussians.scales,
            sigma_world,
        )
        jacobian = torch.zeros(
            n_culled_gaussians, 2, 3, dtype=torch.float32, device=device
        )
        compute_projection_jacobian_cuda(xyz_camera_frame, camera.K, jacobian)

        # compute sigma_image
        sigma_image = torch.zeros(
            n_culled_gaussians, 2, 2, dtype=torch.float32, device=device
        )
        compute_sigma_image_cuda(sigma_world, jacobian, world_T_image, sigma_image)

    return uv, sigma_image, culled_gaussians, xyz_camera_frame
