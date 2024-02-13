import numpy as np
import torch
from torch import nn

TILE_EDGE_LENGTH_PX = 16


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

        # compute top left, top right, bottom left, bottom right of each tile [x, y]
        self.tile_corners = torch.zeros(
            self.tile_count, 4, 2, dtype=torch.int32, device=self.device
        )
        for row in range(self.y_tiles_count):
            for col in range(self.x_tiles_count):
                # top left
                self.tile_corners[row * self.x_tiles_count + col, 0, 0] = (
                    col * self.tile_edge_size
                )
                self.tile_corners[row * self.x_tiles_count + col, 0, 1] = (
                    row * self.tile_edge_size
                )
                # top right
                self.tile_corners[row * self.x_tiles_count + col, 1, 0] = (
                    col + 1
                ) * self.tile_edge_size
                self.tile_corners[row * self.x_tiles_count + col, 1, 1] = (
                    row * self.tile_edge_size
                )
                # bottom left
                self.tile_corners[row * self.x_tiles_count + col, 2, 0] = (
                    col * self.tile_edge_size
                )
                self.tile_corners[row * self.x_tiles_count + col, 2, 1] = (
                    row + 1
                ) * self.tile_edge_size
                # bottom right
                self.tile_corners[row * self.x_tiles_count + col, 3, 0] = (
                    col + 1
                ) * self.tile_edge_size
                self.tile_corners[row * self.x_tiles_count + col, 3, 1] = (
                    row + 1
                ) * self.tile_edge_size
