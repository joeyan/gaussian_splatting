import numpy as np
import torch
from scipy.spatial import KDTree


def inverse_sigmoid(x):
    """
    Inverse of sigmoid activation
    """
    return np.log(x / (1 - x))


def compute_initial_scale_from_sparse_points(
    points, num_neighbors, neighbor_dist_to_scale_factor
):
    """
    Computes the initial gaussian scale from the distance to the nearest points
    """
    points_np = points.cpu().numpy()
    tree = KDTree(points_np)

    n_pts = points_np.shape[0]
    scale = torch.zeros(n_pts, 3, dtype=torch.float32)
    for pt_idx in range(n_pts):
        neighbor_dist_vect, _ = tree.query(
            points_np[pt_idx, :], k=num_neighbors, workers=-1
        )
        # use log since scale has exp activation
        scale[pt_idx, :] = torch.ones(3, dtype=torch.float32) * np.log(
            np.mean(neighbor_dist_vect) * neighbor_dist_to_scale_factor
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
    pts = torch.cat(
        [pts, torch.ones(pts.shape[0], 1, dtype=pts.dtype, device=pts.device)], dim=1
    )
    pts = torch.matmul(transform, pts.unsqueeze(-1)).squeeze(-1)[:, :3]
    return pts.contiguous()
