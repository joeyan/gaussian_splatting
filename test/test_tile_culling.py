import unittest
import torch

from splat_py.utils import transform_points_torch
from splat_py.cuda_autograd_functions import (
    CameraPointProjection,
    ComputeSigmaWorld,
    ComputeProjectionJacobian,
    ComputeConic,
)
from splat_py.structs import Gaussians, Tiles
from splat_py.tile_culling import (
    get_splats,
)
from gaussian_test_data import get_test_data


class TestCulling(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available())
        self.device = torch.device("cuda")
        self.gaussians, self.camera, self.camera_T_world = get_test_data(self.device)

    def test_tile_culling(self):
        near_thresh = 0.3
        cull_mask_padding = 10
        mh_dist = 3.0

        xyz_camera_frame = transform_points_torch(self.gaussians.xyz, self.camera_T_world)
        uv = CameraPointProjection.apply(xyz_camera_frame, self.camera.K)

        # perform frustrum culling
        culling_mask = torch.zeros(
            xyz_camera_frame.shape[0],
            dtype=torch.bool,
            device=self.gaussians.xyz.device,
        )
        culling_mask = culling_mask | (xyz_camera_frame[:, 2] < near_thresh)
        culling_mask = (
            culling_mask
            | (uv[:, 0] < -1 * cull_mask_padding)
            | (uv[:, 0] > self.camera.width + cull_mask_padding)
            | (uv[:, 1] < -1 * cull_mask_padding)
            | (uv[:, 1] > self.camera.height + cull_mask_padding)
        )

        # cull gaussians outside of camera frustrum
        uv = uv[~culling_mask, :]
        xyz_camera_frame = xyz_camera_frame[~culling_mask, :]

        culled_gaussians = Gaussians(
            xyz=self.gaussians.xyz[~culling_mask, :],
            quaternion=self.gaussians.quaternion[~culling_mask, :],
            scale=self.gaussians.scale[~culling_mask, :],
            opacity=torch.sigmoid(
                self.gaussians.opacity[~culling_mask]
            ),  # apply sigmoid activation to opacity
            rgb=self.gaussians.rgb[~culling_mask, :],
        )

        sigma_world = ComputeSigmaWorld.apply(culled_gaussians.quaternion, culled_gaussians.scale)
        J = ComputeProjectionJacobian.apply(xyz_camera_frame, self.camera.K)
        conic = ComputeConic.apply(sigma_world, J, self.camera_T_world)

        # perform tile culling
        tiles = Tiles(self.camera.height, self.camera.width, uv.device)

        sorted_gaussian_idx_by_splat_idx, splat_start_end_idx_by_tile_idx = get_splats(
            uv, tiles, conic, xyz_camera_frame, mh_dist
        )

        # fmt: off
        expected_sorted_gaussian_idx_by_splat_idx = torch.tensor(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0,
            2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0,
            2, 0, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 0,
            2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 1, 2, 1, 2, 1, 2, 1, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2,
            0, 2, 0, 2, 0, 2, 0, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 1, 2,
            1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 0, 2, 0, 2, 0,
            2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 1, 2, 0, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2,
            0, 1, 2, 0, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0,
            2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 1, 2, 0, 1, 2, 1, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0,
            2, 0, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 1, 2, 0, 1, 2,
            0, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 0, 2, 0,
            2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0,
            2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1,
            0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
            1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            device=sorted_gaussian_idx_by_splat_idx.device,
            dtype=sorted_gaussian_idx_by_splat_idx.dtype
        )
        # fmt: on
        self.assertTrue(
            torch.equal(sorted_gaussian_idx_by_splat_idx, expected_sorted_gaussian_idx_by_splat_idx)
        )
        self.assertEqual(splat_start_end_idx_by_tile_idx.shape[0], 1201)


if __name__ == "__main__":
    unittest.main()
