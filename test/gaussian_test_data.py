import torch

from splat_py.structs import Gaussians, Camera


def get_test_gaussians(device):
    xyz = torch.tensor(
        [
            [1.0, 2.0, -4.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, -9.0],
            [1.0, 2.0, 15.0],
            [2.5, -1.0, 4.0],
            [-1.0, -2.0, 10.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    rgb = torch.ones(xyz.shape, dtype=torch.float32, device=device) * 0.5
    rgb[3, :] = torch.tensor([0.5, 0.0, 0.0], dtype=torch.float32, device=device)
    rgb[4, :] = torch.tensor([0.0, 0.5, 0.0], dtype=torch.float32, device=device)
    rgb[5, :] = torch.tensor([0.0, 0.0, 0.5], dtype=torch.float32, device=device)
    opacities = torch.ones(xyz.shape[0], 1, dtype=torch.float32, device=device)
    scales = torch.tensor(
        [
            [0.02, 0.03, 0.04],
            [0.01, 0.05, 0.02],
            [0.09, 0.03, 0.01],
            [1.0, 3.0, 0.1],
            [2.0, 0.2, 0.1],
            [2.0, 1.0, 0.1],
        ],
        dtype=torch.float32,
        device=device,
    )
    # using exp activation
    scales = torch.log(scales)
    quaternions = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.714, -0.002, -0.664, 0.221],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    return Gaussians(xyz, rgb, opacities, scales, quaternions)


def get_test_camera(device):
    # different fx and fy to test computation of gaussian projection
    K = torch.tensor(
        [
            [430.0, 0.0, 320.0],
            [0.0, 410.0, 240.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    return Camera(640, 480, K)


def get_test_world_T_image(device):
    return torch.tensor(
        [
            [0.9999, 0.0089, 0.0073, -0.3283],
            [-0.0106, 0.9568, 0.2905, -1.9260],
            [-0.0044, -0.2906, 0.9568, 2.9581],
            [0.0000, 0.0000, 0.0000, 1.0000],
        ],
        dtype=torch.float32,
        device=device,
    )


def get_test_data(device):
    gaussians = get_test_gaussians(device)
    camera = get_test_camera(device)
    world_T_image = get_test_world_T_image(device)
    return gaussians, camera, world_T_image
