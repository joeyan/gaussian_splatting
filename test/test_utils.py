import math
import unittest
import torch

from splat_py.utils import quaternion_to_rotation_torch, transform_points_torch


class TestUtils(unittest.TestCase):
    def test_quaternion_to_rotation_torch(self):
        q = torch.tensor(
            [1.0, 0.0, 0.0, 0.0, 0.0, math.sqrt(2) / 2, 0.0, math.sqrt(2) / 2]
        )
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

        q = torch.tensor([0.0, math.sqrt(2) / 2, 0.0, math.sqrt(2) / 2]).unsqueeze(
            dim=0
        )
        transform = torch.eye(4)
        transform[:3, :3] = quaternion_to_rotation_torch(q)
        transform[:3, 3] = torch.tensor([1.0, 2.0, 3.0])

        transformed_pts = transform_points_torch(pts, transform)
        self.assertEqual(transformed_pts.shape, expected_pts.shape)

        self.assertTrue(transformed_pts.allclose(expected_pts, atol=1e-6))

        transform_inv = torch.inverse(transform.unsqueeze(dim=0))
        transformed_back_original_pts = transform_points_torch(
            transformed_pts, transform_inv
        )
        self.assertTrue(transformed_back_original_pts.allclose(pts, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
