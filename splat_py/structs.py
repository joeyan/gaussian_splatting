import time
import numpy as np
import torch
from torch import nn

from splat_py.constants import PRINT_DEBUG_TIMING

TILE_EDGE_LENGTH_PX = 16


class SimpleTimer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, type, value, traceback):
        end = time.time()
        if PRINT_DEBUG_TIMING:
            print(f"{self.name}: {(end - self.start) * 1000} ms")


class Image:
    """
    Image and Pose information
    """

    def __init__(
        self,
        image,  # loaded image [HxWx3], 8bit, RGB
        camera_id,  # camera id associated with the image
        world_T_image,  # world to camera transform matrix [4x4]
    ):
        self.image = image
        self.camera_id = camera_id
        self.world_T_image = world_T_image


class Camera:
    """
    Basic Pinhole Camera class
    """

    def __init__(
        self,
        width,  # image width
        height,  # image height
        K,  # camera matrix [3x3]
    ):
        self.width = width
        self.height = height
        self.K = K


class Gaussians(nn.Module):
    """
    Contains all mutable gaussian parameters
    """

    def __init__(
        self,
        xyz,  # Nx3 [x, y, z]
        rgb,  # Nx3 [r, g, b] normalized to [0, 1]
        opacities,  # Nx1 [opacity] from [0, 1]
        scales,  # Nx3 [sx, sy, sz]
        quaternions,  # Nx4 [qw, qx, qy, qz]
    ):
        super().__init__()
        self.xyz = xyz
        self.rgb = rgb
        self.opacities = opacities
        self.scales = scales
        self.quaternions = quaternions

    def filter_in_place(self, keep_mask):
        self.xyz = torch.nn.Parameter(self.xyz[keep_mask, :])
        self.rgb = torch.nn.Parameter(self.rgb[keep_mask, :])
        self.opacities = torch.nn.Parameter(self.opacities[keep_mask])
        self.scales = torch.nn.Parameter(self.scales[keep_mask, :])
        self.quaternions = torch.nn.Parameter(self.quaternions[keep_mask, :])


class Tiles:
    """
    Tiles for rasterization
    """

    def __init__(self, image_height, image_width, device):
        self.image_height = image_height
        self.image_width = image_width
        self.device = device
        self.tile_edge_size = TILE_EDGE_LENGTH_PX

        # Need to round up to the nearest multiple of TILE_EDGE_LENGTH_PX to ensure all pixels are covered
        self.image_height_padded = int(
            np.ceil(image_height / self.tile_edge_size) * self.tile_edge_size
        )
        self.image_width_padded = int(
            np.ceil(image_width / self.tile_edge_size) * self.tile_edge_size
        )

        self.y_tiles_count = int(self.image_height_padded / self.tile_edge_size)
        self.x_tiles_count = int(self.image_width_padded / self.tile_edge_size)
        self.tile_count = self.y_tiles_count * self.x_tiles_count
