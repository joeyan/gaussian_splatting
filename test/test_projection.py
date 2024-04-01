import unittest
import torch

from splat_py.cuda_autograd_functions import (
    CameraPointProjection,
    ComputeSigmaWorld,
    ComputeProjectionJacobian,
    ComputeConic,
)
from splat_py.utils import transform_points_torch

from gaussian_test_data import get_test_data


class ProjectionTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available())
        self.device = torch.device("cuda")
        self.gaussians, self.camera, self.world_T_image = get_test_data(self.device)

    def test_project_points(self):
        xyz_camera_frame = transform_points_torch(self.gaussians.xyz, self.world_T_image)

        self.assertAlmostEqual(xyz_camera_frame[0, 0].item(), 0.6602, places=4)
        self.assertAlmostEqual(xyz_camera_frame[0, 1].item(), -1.1849998, places=4)
        self.assertAlmostEqual(xyz_camera_frame[0, 2].item(), -1.4546999, places=4)
        self.assertAlmostEqual(xyz_camera_frame[1, 0].item(), 3.7595997, places=4)
        self.assertAlmostEqual(xyz_camera_frame[1, 1].item(), 4.5586, places=4)
        self.assertAlmostEqual(xyz_camera_frame[1, 2].item(), 7.2283, places=4)

        uv = CameraPointProjection.apply(xyz_camera_frame, self.camera.K)

        self.assertEqual(uv.shape, (6, 2))
        self.assertAlmostEqual(uv[0, 0].item(), 124.849106, places=4)
        self.assertAlmostEqual(uv[0, 1].item(), 573.9863, places=4)
        self.assertAlmostEqual(uv[1, 0].item(), 543.6526, places=4)
        self.assertAlmostEqual(uv[1, 1].item(), 498.57062, places=4)

        # perform frustrum culling
        # (TODO) move frustrum culling to function
        culling_mask = torch.zeros(
            xyz_camera_frame.shape[0],
            dtype=torch.bool,
            device=self.device,
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

        self.assertEqual(uv.shape, (6, 2))
        self.assertEqual(xyz_camera_frame.shape, (6, 3))
        self.assertEqual(culling_mask.shape, (6,))

        self.assertTrue(
            torch.all(
                culling_mask
                == torch.tensor([True, True, True, False, False, False], device=self.device)
            )
        )

    def test_compute_sigma_world(self):
        sigma_world = ComputeSigmaWorld.apply(self.gaussians.quaternions, self.gaussians.scales)

        self.assertEqual(sigma_world.shape, (6, 3, 3))
        # check first sigma_world
        self.assertAlmostEqual(sigma_world[0, 0, 0].item(), 0.0004, places=4)
        self.assertAlmostEqual(sigma_world[0, 0, 1].item(), 0.0, places=4)
        self.assertAlmostEqual(sigma_world[0, 0, 2].item(), 0.0, places=4)

        self.assertAlmostEqual(sigma_world[0, 1, 0].item(), 0.0, places=4)
        self.assertAlmostEqual(sigma_world[0, 1, 1].item(), 0.0009, places=4)
        self.assertAlmostEqual(sigma_world[0, 1, 2].item(), 0.0, places=4)

        self.assertAlmostEqual(sigma_world[0, 2, 0].item(), 0.0, places=4)
        self.assertAlmostEqual(sigma_world[0, 2, 1].item(), 0.0, places=4)
        self.assertAlmostEqual(sigma_world[0, 2, 2].item(), 0.0016, places=4)

        # sigma world
        self.assertAlmostEqual(sigma_world[4, 0, 0].item(), 0.01454808, places=4)
        self.assertAlmostEqual(sigma_world[4, 0, 1].item(), 0.01702517, places=4)
        self.assertAlmostEqual(sigma_world[4, 0, 2].item(), 0.07868834, places=4)
        self.assertAlmostEqual(sigma_world[4, 1, 0].item(), 0.01702517, places=4)
        self.assertAlmostEqual(sigma_world[4, 1, 1].item(), 0.4389012, places=4)
        self.assertAlmostEqual(sigma_world[4, 1, 2].item(), 1.1959752, places=4)
        self.assertAlmostEqual(sigma_world[4, 2, 0].item(), 0.07868834, places=4)
        self.assertAlmostEqual(sigma_world[4, 2, 1].item(), 1.1959752, places=4)
        self.assertAlmostEqual(sigma_world[4, 2, 2].item(), 3.5965507, places=4)

    def test_compute_projection_jacobian(self):
        xyz_camera_frame = transform_points_torch(self.gaussians.xyz, self.world_T_image)

        jacobian = ComputeProjectionJacobian.apply(xyz_camera_frame, self.camera.K)

        self.assertEqual(jacobian.shape, (6, 2, 3))
        self.assertAlmostEqual(jacobian[0, 0, 0].item(), -295.5936, places=4)
        self.assertAlmostEqual(jacobian[0, 0, 1].item(), 0.0, places=4)
        self.assertAlmostEqual(jacobian[0, 0, 2].item(), -134.1520, places=4)
        self.assertAlmostEqual(jacobian[0, 1, 0].item(), 0.0, places=4)
        self.assertAlmostEqual(jacobian[0, 1, 1].item(), -281.8451, places=4)
        self.assertAlmostEqual(jacobian[0, 1, 2].item(), 229.5912, places=4)

    def test_compute_conic(self):
        # compute inputs (tested in previous tests)
        sigma_world = ComputeSigmaWorld.apply(self.gaussians.quaternions, self.gaussians.scales)
        xyz_camera_frame = transform_points_torch(self.gaussians.xyz, self.world_T_image)
        jacobian = ComputeProjectionJacobian.apply(xyz_camera_frame, self.camera.K)

        # compute conic
        conic = ComputeConic.apply(sigma_world, jacobian, self.world_T_image)

        self.assertEqual(conic.shape, (6, 3))
        self.assertAlmostEqual(conic[3, 0].item(), 664.28760, places=4)
        self.assertAlmostEqual(conic[3, 1].item(), 254.81781, places=4)
        self.assertAlmostEqual(conic[3, 2].item(), 5761.8906, places=4)


if __name__ == "__main__":
    unittest.main()
