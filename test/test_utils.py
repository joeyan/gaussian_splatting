import math
import unittest
import torch

from splat_py.utils import (
    quaternion_to_rotation_torch,
    transform_points_torch,
    compute_rays,
    compute_rays_in_world_frame,
)
from gaussian_test_data import get_test_camera, get_test_camera_T_world


class TestUtils(unittest.TestCase):
    def test_quaternion_to_rotation_torch(self):
        q = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, math.sqrt(2) / 2, 0.0, math.sqrt(2) / 2])
        q = q.reshape(-1, 4)
        R = quaternion_to_rotation_torch(q)

        self.assertEqual(R.shape, (2, 3, 3))
        # transpose/inverse each rotation matrix in the 3D tensor
        # R * R_inv = I
        R_inv = torch.transpose(R, 1, 2)
        eye_tensor = torch.eye(3).repeat(2, 1, 1)
        self.assertTrue(torch.allclose(torch.bmm(R, R_inv), eye_tensor, atol=1e-6))

    def test_transform_points_torch(self):
        pts = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        pts = pts.reshape(-1, 3)
        expected_pts = torch.tensor([4.0, 0.0, 4.0, 7.0, -3.0, 7.0, 10.0, -6.0, 10.0])
        expected_pts = expected_pts.reshape(-1, 3)

        q = torch.tensor([0.0, math.sqrt(2) / 2, 0.0, math.sqrt(2) / 2]).unsqueeze(dim=0)
        transform = torch.eye(4)
        transform[:3, :3] = quaternion_to_rotation_torch(q)
        transform[:3, 3] = torch.tensor([1.0, 2.0, 3.0])

        transformed_pts = transform_points_torch(pts, transform)
        self.assertEqual(transformed_pts.shape, expected_pts.shape)

        self.assertTrue(transformed_pts.allclose(expected_pts, atol=1e-6))

        transform_inv = torch.inverse(transform.unsqueeze(dim=0))
        transformed_back_original_pts = transform_points_torch(transformed_pts, transform_inv)
        self.assertTrue(transformed_back_original_pts.allclose(pts, atol=1e-6))

    def test_compute_rays_camera_frame(self):
        # get test data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        camera = get_test_camera(device)

        # compute rays
        rays = compute_rays(camera)
        self.assertEqual(rays.shape, (640 * 480, 3))
        rays = rays.reshape(camera.height, camera.width, 3)
        self.assertEqual(rays.shape, (480, 640, 3))

        # check some values
        self.assertAlmostEqual(rays[0, 0, 0].item(), -0.5403921008110046)
        self.assertAlmostEqual(rays[0, 0, 1].item(), -0.4250645041465759)
        self.assertAlmostEqual(rays[0, 0, 2].item(), 0.7261518836021423)

        self.assertAlmostEqual(rays[240, 320, 0].item(), 0.0)
        self.assertAlmostEqual(rays[240, 320, 1].item(), 0.0)
        self.assertAlmostEqual(rays[240, 320, 2].item(), 1.0)

        self.assertAlmostEqual(rays[0, 639, 0].item(), 0.5391948819160461)
        self.assertAlmostEqual(rays[0, 639, 1].item(), -0.425452321767807)
        self.assertAlmostEqual(rays[0, 639, 2].item(), 0.7268144488334656)

    def test_compute_rays_world_frame(self):
        # get test data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        camera = get_test_camera(device)
        camera_T_world = get_test_camera_T_world(device)
        rays = compute_rays_in_world_frame(camera, camera_T_world)
        self.assertEqual(rays.shape, (480, 640, 3))

        # check some values
        self.assertAlmostEqual(rays[0, 0, 0].item(), -0.5390445590019226)
        self.assertAlmostEqual(rays[0, 0, 1].item(), -0.6224945187568665)
        self.assertAlmostEqual(rays[0, 0, 2].item(), 0.5673900842666626)

        self.assertAlmostEqual(rays[240, 320, 0].item(), -0.004399406723678112)
        self.assertAlmostEqual(rays[240, 320, 1].item(), -0.2905626893043518)
        self.assertAlmostEqual(rays[240, 320, 2].item(), 0.9568459391593933)

        self.assertAlmostEqual(rays[0, 639, 0].item(), 0.540492832660675)
        self.assertAlmostEqual(rays[0, 639, 1].item(), -0.6134769916534424)
        self.assertAlmostEqual(rays[0, 639, 2].item(), 0.5757721662521362)


if __name__ == "__main__":
    unittest.main()
