import numpy as np
import torch
from scipy.spatial import KDTree
from torch.distributions.multivariate_normal import MultivariateNormal

from splat_py.constants import MAX_INITIAL_SCALE


def inverse_sigmoid(x):
    """
    Inverse of sigmoid activation
    """
    clipped = np.clip(x, 1e-4, 1 - 1e-4)
    return np.log(clipped / (1.0 - (clipped)))


def inverse_sigmoid_torch(x):
    clipped = torch.clip(x, 1e-4, 1 - 1e-4)
    return torch.log(clipped / (1.0 - (clipped)))


def sample_pdf(xyz, sigma_world):
    pdf = MultivariateNormal(xyz, sigma_world)
    mean_1 = pdf.sample()
    mean_2 = pdf.sample()
    return mean_1, mean_2


def compute_initial_scale_from_sparse_points(points, num_neighbors, neighbor_dist_to_scale_factor):
    """
    Computes the initial gaussian scale from the distance to the nearest points
    """
    points_np = points.cpu().numpy()
    tree = KDTree(points_np)

    n_pts = points_np.shape[0]
    scale = torch.zeros(n_pts, 3, dtype=torch.float32)
    for pt_idx in range(n_pts):
        neighbor_dist_vect, _ = tree.query(points_np[pt_idx, :], k=num_neighbors, workers=-1)
        initial_scale = min(np.mean(neighbor_dist_vect), MAX_INITIAL_SCALE)
        # use log since scale has exp activation
        scale[pt_idx, :] = torch.ones(3, dtype=torch.float32) * np.log(
            initial_scale * neighbor_dist_to_scale_factor
        )
    return scale


def quaternion_to_rotation_torch(q):
    """'
    Convert tensor of normalized quaternions [N, 4] in [w, x, y, z] format to rotation matrices
    [N, 3, 3]
    """
    rot = [
        1 - 2 * q[:, 2] ** 2 - 2 * q[:, 3] ** 2,
        2 * q[:, 1] * q[:, 2] - 2 * q[:, 0] * q[:, 3],
        2 * q[:, 3] * q[:, 1] + 2 * q[:, 0] * q[:, 2],
        2 * q[:, 1] * q[:, 2] + 2 * q[:, 0] * q[:, 3],
        1 - 2 * q[:, 1] ** 2 - 2 * q[:, 3] ** 2,
        2 * q[:, 2] * q[:, 3] - 2 * q[:, 0] * q[:, 1],
        2 * q[:, 3] * q[:, 1] - 2 * q[:, 0] * q[:, 2],
        2 * q[:, 2] * q[:, 3] + 2 * q[:, 0] * q[:, 1],
        1 - 2 * q[:, 1] ** 2 - 2 * q[:, 2] ** 2,
    ]
    rot = torch.stack(rot, dim=1).reshape(-1, 3, 3)
    return rot


def transform_points_torch(pts, transform):  # N x 3  # N x 4 x 4
    """
    Transform points by a 4x4 matrix
    """
    pts = torch.cat([pts, torch.ones(pts.shape[0], 1, dtype=pts.dtype, device=pts.device)], dim=1)
    transformed_pts = torch.matmul(transform, pts.unsqueeze(-1)).squeeze(-1)[:, :3]

    if torch.isnan(transformed_pts).any():
        print("NaN in transform_points_torch")
        filtered_tensor = pts[torch.any(transformed_pts.isnan(), dim=1)]
        print(filtered_tensor.detach().cpu().numpy())

    return transformed_pts.contiguous()


def compute_rays(camera):
    """
    Compute rays in camera space
    """
    # grid of uv coordinates
    u = torch.linspace(
        0, camera.width - 1, camera.width, dtype=camera.K.dtype, device=camera.K.device
    )
    v = torch.linspace(
        0,
        camera.height - 1,
        camera.height,
        dtype=camera.K.dtype,
        device=camera.K.device,
    )

    # use (v, u) order to preserve row-major order
    v, u = torch.meshgrid(v, u, indexing="ij")
    v = v.flatten()
    u = u.flatten()

    K = camera.K
    # Inverse pinhole projection
    # fx * x/z + cx = u => x/z = (u - cx) / fx, z = 1
    # fy * y/z + cy = v => y/z = (v - cy) / fy, z = 1
    ray_dir = torch.stack(
        [
            (u - K[0, 2]) / K[0, 0],
            (v - K[1, 2]) / K[1, 1],
            torch.ones_like(u),
        ],
        dim=-1,
    )
    ray_dir = ray_dir / torch.norm(ray_dir, dim=1, keepdim=True)
    return ray_dir


def compute_rays_in_world_frame(camera, camera_T_world):
    """
    Compute rays in world space
    """
    rays = compute_rays(camera)
    # transform rays to world space
    world_T_camera = torch.inverse(camera_T_world)
    rays = (world_T_camera[:3, :3] @ rays.T).T
    rays = rays / torch.norm(rays, dim=1, keepdim=True)
    rays = rays.reshape(camera.height, camera.width, 3)
    rays = rays.contiguous()
    return rays
