import numpy as np
import torch
from scipy.spatial import KDTree
from torch.distributions.multivariate_normal import MultivariateNormal


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
    transformed_pts = torch.matmul(transform, pts.unsqueeze(-1)).squeeze(-1)[:, :3]

    if torch.isnan(transformed_pts).any():
        print("NaN in transform_points_torch")
        filtered_tensor = pts[torch.any(transformed_pts.isnan(), dim=1)]
        print(filtered_tensor.detach().cpu().numpy())

    return transformed_pts.contiguous()
