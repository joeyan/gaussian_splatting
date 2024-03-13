import unittest
import torch

from splat_py.projection import (
    compute_sigma_world,
    project_points,
    compute_projection_jacobian,
    compute_sigma_image,
)
from splat_cuda import (
    compute_sigma_world_cuda,
    compute_projection_jacobian_cuda,
    compute_sigma_image_cuda,
)
from gaussian_test_data import get_test_data


class ProjectionTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available())
        self.device = torch.device("cuda")
        self.gaussians, self.camera, self.world_T_image = get_test_data(self.device)

    def test_compute_sigma_world(self):
        sigma_world_py = compute_sigma_world(self.gaussians)
        self.assertEqual(sigma_world_py.shape, (6, 3, 3))

        sigma_world_cuda = torch.zeros(
            (6, 3, 3), dtype=torch.float32, device=torch.device("cuda")
        )
        compute_sigma_world_cuda(
            self.gaussians.quaternions,
            self.gaussians.scales,
            sigma_world_cuda,
        )
        self.assertTrue(torch.allclose(sigma_world_py, sigma_world_cuda, atol=1e-6))

        # check first sigma_world
        self.assertAlmostEqual(sigma_world_py[0, 0, 0].item(), 0.0004, places=4)
        self.assertAlmostEqual(sigma_world_py[0, 0, 1].item(), 0.0, places=4)
        self.assertAlmostEqual(sigma_world_py[0, 0, 2].item(), 0.0, places=4)

        self.assertAlmostEqual(sigma_world_py[0, 1, 0].item(), 0.0, places=4)
        self.assertAlmostEqual(sigma_world_py[0, 1, 1].item(), 0.0009, places=4)
        self.assertAlmostEqual(sigma_world_py[0, 1, 2].item(), 0.0, places=4)

        self.assertAlmostEqual(sigma_world_py[0, 2, 0].item(), 0.0, places=4)
        self.assertAlmostEqual(sigma_world_py[0, 2, 1].item(), 0.0, places=4)
        self.assertAlmostEqual(sigma_world_py[0, 2, 2].item(), 0.0016, places=4)

        # sigma world
        self.assertAlmostEqual(sigma_world_py[4, 0, 0].item(), 0.01454808, places=4)
        self.assertAlmostEqual(sigma_world_py[4, 0, 1].item(), 0.01702517, places=4)
        self.assertAlmostEqual(sigma_world_py[4, 0, 2].item(), 0.07868834, places=4)
        self.assertAlmostEqual(sigma_world_py[4, 1, 0].item(), 0.01702517, places=4)
        self.assertAlmostEqual(sigma_world_py[4, 1, 1].item(), 0.4389012, places=4)
        self.assertAlmostEqual(sigma_world_py[4, 1, 2].item(), 1.1959752, places=4)
        self.assertAlmostEqual(sigma_world_py[4, 2, 0].item(), 0.07868834, places=4)
        self.assertAlmostEqual(sigma_world_py[4, 2, 1].item(), 1.1959752, places=4)
        self.assertAlmostEqual(sigma_world_py[4, 2, 2].item(), 3.5965507, places=4)

    def test_project_points(self):
        uv, xyz_camera_frame, culling_mask = project_points(
            self.world_T_image, self.camera, self.gaussians
        )
        self.assertEqual(uv.shape, (6, 2))
        self.assertEqual(xyz_camera_frame.shape, (6, 3))
        self.assertEqual(culling_mask.shape, (6,))

        self.assertTrue(
            torch.all(
                culling_mask
                == torch.tensor(
                    [True, True, True, False, False, False], device=self.device
                )
            )
        )

        self.assertAlmostEqual(uv[0, 0].item(), 124.849106, places=4)
        self.assertAlmostEqual(uv[0, 1].item(), 573.9863, places=4)
        self.assertAlmostEqual(uv[1, 0].item(), 543.6526, places=4)
        self.assertAlmostEqual(uv[1, 1].item(), 498.57062, places=4)

        self.assertAlmostEqual(xyz_camera_frame[0, 0].item(), 0.6602, places=4)
        self.assertAlmostEqual(xyz_camera_frame[0, 1].item(), -1.1849998, places=4)
        self.assertAlmostEqual(xyz_camera_frame[0, 2].item(), -1.4546999, places=4)
        self.assertAlmostEqual(xyz_camera_frame[1, 0].item(), 3.7595997, places=4)
        self.assertAlmostEqual(xyz_camera_frame[1, 1].item(), 4.5586, places=4)
        self.assertAlmostEqual(xyz_camera_frame[1, 2].item(), 7.2283, places=4)

    def test_compute_projection_jacobian(self):
        _, xyz_camera_frame, _ = project_points(
            self.world_T_image, self.camera, self.gaussians
        )
        jacobian_py = compute_projection_jacobian(xyz_camera_frame, self.camera)
        self.assertEqual(jacobian_py.shape, (6, 2, 3))

        self.assertAlmostEqual(jacobian_py[0, 0, 0].item(), -295.5936, places=4)
        self.assertAlmostEqual(jacobian_py[0, 0, 1].item(), 0.0, places=4)
        self.assertAlmostEqual(jacobian_py[0, 0, 2].item(), -134.1520, places=4)
        self.assertAlmostEqual(jacobian_py[0, 1, 0].item(), 0.0, places=4)
        self.assertAlmostEqual(jacobian_py[0, 1, 1].item(), -281.8451, places=4)
        self.assertAlmostEqual(jacobian_py[0, 1, 2].item(), 229.5912, places=4)

        jacobian_cuda = torch.zeros(6, 2, 3, dtype=torch.float32, device=self.device)
        compute_projection_jacobian_cuda(xyz_camera_frame, self.camera.K, jacobian_cuda)
        self.assertTrue(torch.allclose(jacobian_py, jacobian_cuda, atol=1e-6))

    def test_compute_sigma_image(self):
        _, xyz_camera_frame, _ = project_points(
            self.world_T_image, self.camera, self.gaussians
        )
        sigma_image_py = compute_sigma_image(
            self.gaussians, xyz_camera_frame, self.camera, self.world_T_image
        )
        self.assertEqual(sigma_image_py.shape, (6, 2, 2))

        self.assertAlmostEqual(sigma_image_py[3, 0, 0].item(), 664.28760, places=4)
        self.assertAlmostEqual(sigma_image_py[3, 0, 1].item(), 127.40891, places=4)
        self.assertAlmostEqual(sigma_image_py[3, 1, 0].item(), 127.40890, places=4)
        self.assertAlmostEqual(sigma_image_py[3, 1, 1].item(), 5761.8906, places=4)

        # compute inputs (tested in previous tests)
        sigma_world_cuda = torch.zeros(
            (6, 3, 3), dtype=torch.float32, device=torch.device("cuda")
        )
        compute_sigma_world_cuda(
            self.gaussians.quaternions,
            self.gaussians.scales,
            sigma_world_cuda,
        )
        jacobian_cuda = torch.zeros(6, 2, 3, dtype=torch.float32, device=self.device)
        compute_projection_jacobian_cuda(xyz_camera_frame, self.camera.K, jacobian_cuda)

        # compute sigma_image
        sigma_image_cuda = torch.zeros(6, 2, 2, dtype=torch.float32, device=self.device)
        compute_sigma_image_cuda(
            sigma_world_cuda, jacobian_cuda, self.world_T_image, sigma_image_cuda
        )

        self.assertTrue(torch.allclose(sigma_image_py, sigma_image_cuda, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
