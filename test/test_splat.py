import unittest
import numpy as np
import torch
from splat import camera_projection, render_tiles
from structs import Gaussians, Camera
import cv2


class TestSplatFull(unittest.TestCase):
    def setup_gaussians(self):
        xyz = torch.tensor(
            [
                1.0,
                2.0,
                -4.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                -9.0,
                1.0,
                2.0,
                15.0,
                5.0,
                1.0,
                4.0,
                -1.0,
                -2.0,
                10.0,
            ],
            dtype=torch.float32,
            device=self.device,
        ).reshape(-1, 3)
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
                0.02,
                0.03,
                0.04,
                0.01,
                0.05,
                0.02,
                0.09,
                0.03,
                0.01,
                1.0,
                3.0,
                0.1,
                2.0,
                0.2,
                0.1,
                2.0,
                1.0,
                0.1,
            ],
            dtype=torch.float32,
            device=self.device,
        ).reshape(-1, 3)
        # using exp activation
        scales = torch.log(scales)
        quaternions = torch.tensor(
            [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.9238795,
                0.0,
                0.0,
                0.3826834,  # 45 degree rotation around z
                1.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=torch.float32,
            device=self.device,
        ).reshape(-1, 4)
        self.gaussians = Gaussians(xyz, rgb, opacities, scales, quaternions)

    def setup_camera(self):
        # different fx and fy to test computation of gaussian projection
        K = torch.tensor(
            [430.0, 0.0, 320.0, 0.0, 410.0, 240.0, 0.0, 0.0, 1.0],
            dtype=torch.float32,
            device=self.device,
        ).reshape(3, 3)
        self.camera = Camera(640, 480, K)

    def setup_world_T_image(self):
        self.world_T_image = torch.tensor(
            [
                0.9999,
                0.0089,
                0.0073,
                -0.3283,
                -0.0106,
                0.9568,
                0.2905,
                -1.9260,
                -0.0044,
                -0.2906,
                0.9568,
                2.9581,
                0.0000,
                0.0000,
                0.0000,
                1.0000,
            ],
            dtype=torch.float32,
            device=self.device,
        ).reshape(4, 4)

    def setUp(self):
        self.device = torch.device("cpu")

        self.setup_gaussians()
        print(self.gaussians.xyz.shape)

        self.setup_camera()
        self.setup_world_T_image()

    def test_camera_projection(self):
        with torch.no_grad():
            uv, gaussian_2d, culling_mask = camera_projection(
                self.world_T_image, self.camera, self.gaussians
            )
            image = render_tiles(
                uv, self.gaussians, gaussian_2d, culling_mask, self.camera
            )

            self.assertAlmostEqual(
                image[340, 348, 0].detach().cpu().numpy(), 0.477, places=3
            )
            self.assertAlmostEqual(
                image[340, 348, 1].detach().cpu().numpy(), 0.0, places=3
            )
            self.assertAlmostEqual(
                image[340, 348, 2].detach().cpu().numpy(), 0.0, places=3
            )

            image = image.clip(0, 1).detach().cpu().numpy()
            cv2.imwrite("test_splat.png", (image * 255).astype(np.uint8)[..., ::-1])


if __name__ == "__main__":
    unittest.main()
