import cv2
import numpy as np
import torch
import unittest

from splat_py.cuda_autograd_functions import (
    CameraPointProjection,
    ComputeSigmaWorld,
    ComputeProjectionJacobian,
    ComputeSigmaImage,
    RenderImage,
)
from splat_py.structs import Gaussians, Tiles
from splat_py.tile_culling import (
    match_gaussians_to_tiles_gpu,
    sort_gaussians,
)
from splat_py.utils import transform_points_torch

from gaussian_test_data import get_test_data


class TestSplatFull(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available())
        self.device = torch.device("cuda")
        self.gaussians, self.camera, self.world_T_image = get_test_data(self.device)

    def test_splat_gpu(self):
        xyz_camera_frame = transform_points_torch(
            self.gaussians.xyz, self.world_T_image
        )
        uv = CameraPointProjection.apply(xyz_camera_frame, self.camera.K)

        # perform frustrum culling
        culling_mask = torch.zeros(
            xyz_camera_frame.shape[0],
            dtype=torch.bool,
            device=self.gaussians.xyz.device,
        )
        near_thresh = 0.3
        culling_mask = culling_mask | (xyz_camera_frame[:, 2] < near_thresh)
        culling_mask = (
            culling_mask
            | (uv[:, 0] < 0)
            | (uv[:, 0] > self.camera.width)
            | (uv[:, 1] < 0)
            | (uv[:, 1] > self.camera.height)
        )

        # cull gaussians outside of camera frustrum
        uv = uv[~culling_mask, :]
        xyz_camera_frame = xyz_camera_frame[~culling_mask, :]

        culled_gaussians = Gaussians(
            xyz=self.gaussians.xyz[~culling_mask, :],
            quaternions=self.gaussians.quaternions[~culling_mask, :],
            scales=self.gaussians.scales[~culling_mask, :],
            opacities=self.gaussians.opacities[~culling_mask],
            rgb=self.gaussians.rgb[~culling_mask, :],
        )

        sigma_world = ComputeSigmaWorld.apply(
            culled_gaussians.quaternions, culled_gaussians.scales
        )
        J = ComputeProjectionJacobian.apply(xyz_camera_frame, self.camera.K)
        sigma_image = ComputeSigmaImage.apply(sigma_world, J, self.world_T_image)

        # perform tile culling
        tiles = Tiles(self.camera.height, self.camera.width, uv.device)
        (
            gaussian_idx_by_splat_idx,
            splat_start_end_idx_by_tile_idx,
            tile_idx_by_splat_idx,
        ) = match_gaussians_to_tiles_gpu(uv, tiles, sigma_image, mh_dist=3.0)

        sorted_gaussian_idx_by_splat_idx = sort_gaussians(
            xyz_camera_frame, gaussian_idx_by_splat_idx, tile_idx_by_splat_idx
        )
        image = RenderImage.apply(
            culled_gaussians.rgb,
            culled_gaussians.opacities,
            uv,
            sigma_image,
            splat_start_end_idx_by_tile_idx,
            sorted_gaussian_idx_by_splat_idx,
            torch.tensor([self.camera.height, self.camera.width], device=uv.device),
        )

        # near red gaussian center
        self.assertAlmostEqual(image[340, 348, 0].item(), 0.47702518, places=5)
        self.assertAlmostEqual(image[340, 348, 1].item(), 0.0, places=5)
        self.assertAlmostEqual(image[340, 348, 2].item(), 0.0, places=5)

        # overlap of red and blue gaussian, blue is in front of red
        self.assertAlmostEqual(image[200, 348, 0].item(), 0.03330786, places=5)
        self.assertAlmostEqual(image[200, 348, 1].item(), 0.0, places=5)
        self.assertAlmostEqual(image[200, 348, 2].item(), 0.26757469, places=5)
        image = image.clip(0, 1).detach().cpu().numpy()
        cv2.imwrite("test_splat.png", (image * 255).astype(np.uint8)[..., ::-1])


if __name__ == "__main__":
    unittest.main()
