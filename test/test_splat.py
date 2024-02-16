import unittest
import numpy as np
import torch

from projection import camera_projection
from splat import render_tiles, render_tiles_gpu
from structs import Gaussians, Camera
import cv2


class TestSplatFull(unittest.TestCase):
    def setup_gaussians(self):
        xyz = torch.tensor(
            [
                [1.0, 2.0, -4.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, -9.0],
                [1.0, 2.0, 15.0],
                [5.0, 1.0, 4.0],
                [-1.0, -2.0, 10.0],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        rgb = torch.ones(xyz.shape, dtype=torch.float32, device=self.device) * 0.5
        rgb[3, :] = torch.tensor(
            [0.5, 0.0, 0.0], dtype=torch.float32, device=self.device
        )
        rgb[4, :] = torch.tensor(
            [0.0, 0.5, 0.0], dtype=torch.float32, device=self.device
        )
        rgb[5, :] = torch.tensor(
            [0.0, 0.0, 0.5], dtype=torch.float32, device=self.device
        )
        opacities = torch.ones(xyz.shape[0], 1, dtype=torch.float32, device=self.device)
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
            device=self.device,
        )
        # using exp activation
        scales = torch.log(scales)
        quaternions = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.9238795, 0.0, 0.0, 0.3826834],  # 45 degree rotation around z
                [1.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self.gaussians = Gaussians(xyz, rgb, opacities, scales, quaternions)

    def setup_camera(self):
        # different fx and fy to test computation of gaussian projection
        K = torch.tensor(
            [
                [430.0, 0.0, 320.0],
                [0.0, 410.0, 240.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self.camera = Camera(640, 480, K)

    def setup_world_T_image(self):
        self.world_T_image = torch.tensor(
            [
                [0.9999, 0.0089, 0.0073, -0.3283],
                [-0.0106, 0.9568, 0.2905, -1.9260],
                [-0.0044, -0.2906, 0.9568, 2.9581],
                [0.0000, 0.0000, 0.0000, 1.0000],
            ],
            dtype=torch.float32,
            device=self.device,
        )

    def setup_gpu(self):
        self.assertTrue(torch.cuda.is_available())
        self.device = torch.device("cuda")

        self.setup_gaussians()
        self.setup_camera()
        self.setup_world_T_image()

    def setup_cpu(self):
        self.device = torch.device("cpu")

        self.setup_gaussians()
        self.setup_camera()
        self.setup_world_T_image()

    def test_splat_cpu(self):
        self.setup_cpu()

        with torch.no_grad():
            uv, gaussian_2d, culled_gaussians = camera_projection(
                self.world_T_image, self.camera, self.gaussians
            )
            image = render_tiles(uv, culled_gaussians, gaussian_2d, self.camera)

            # near red gaussian center
            self.assertAlmostEqual(
                image[340, 348, 0].detach().cpu().numpy(), 0.47702518, places=6
            )
            self.assertAlmostEqual(
                image[340, 348, 1].detach().cpu().numpy(), 0.0, places=6
            )
            self.assertAlmostEqual(
                image[340, 348, 2].detach().cpu().numpy(), 0.0, places=6
            )

            # overlap of red and blue gaussian
            self.assertAlmostEqual(
                image[200, 348, 0].detach().cpu().numpy(), 0.07165284, places=6
            )
            self.assertAlmostEqual(
                image[200, 348, 1].detach().cpu().numpy(), 0.0, places=6
            )
            self.assertAlmostEqual(
                image[200, 348, 2].detach().cpu().numpy(), 0.22922967, places=6
            )

            image = image.clip(0, 1).detach().cpu().numpy()
            cv2.imwrite("test_splat_cpu.png", (image * 255).astype(np.uint8)[..., ::-1])

    def test_splat_gpu(self):
        self.setup_gpu()

        with torch.no_grad():
            uv, gaussian_2d, culled_gaussians = camera_projection(
                self.world_T_image, self.camera, self.gaussians
            )
            image = render_tiles_gpu(uv, culled_gaussians, gaussian_2d, self.camera)
            # near red gaussian center
            self.assertAlmostEqual(
                image[340, 348, 0].detach().cpu().numpy(), 0.47702518, places=5
            )
            self.assertAlmostEqual(
                image[340, 348, 1].detach().cpu().numpy(), 0.0, places=5
            )
            self.assertAlmostEqual(
                image[340, 348, 2].detach().cpu().numpy(), 0.0, places=5
            )

            # overlap of red and blue gaussian
            self.assertAlmostEqual(
                image[200, 348, 0].detach().cpu().numpy(), 0.07165284, places=5
            )
            self.assertAlmostEqual(
                image[200, 348, 1].detach().cpu().numpy(), 0.0, places=5
            )
            self.assertAlmostEqual(
                image[200, 348, 2].detach().cpu().numpy(), 0.22922967, places=5
            )
            image = image.clip(0, 1).detach().cpu().numpy()
            cv2.imwrite("test_splat_gpu.png", (image * 255).astype(np.uint8)[..., ::-1])


if __name__ == "__main__":
    unittest.main()
