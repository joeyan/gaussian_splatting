import unittest
import numpy as np
import torch
import cv2

from projection import project_and_cull
from splat import render_tiles, render_tiles_gpu
from gaussian_test_data import get_test_data


class TestSplatFull(unittest.TestCase):
    def setup_gpu(self):
        self.assertTrue(torch.cuda.is_available())
        self.device = torch.device("cuda")
        self.gaussians, self.camera, self.world_T_image = get_test_data(self.device)

    def setup_cpu(self):
        self.device = torch.device("cpu")
        self.gaussians, self.camera, self.world_T_image = get_test_data(self.device)

    def test_splat_cpu(self):
        self.setup_cpu()

        with torch.no_grad():
            uv, gaussian_2d, culled_gaussians = project_and_cull(
                self.world_T_image, self.camera, self.gaussians
            )
            image = render_tiles(uv, culled_gaussians, gaussian_2d, self.camera)

            # near red gaussian center
            self.assertAlmostEqual(image[340, 348, 0].item(), 0.47702518, places=6)
            self.assertAlmostEqual(image[340, 348, 1].item(), 0.0, places=6)
            self.assertAlmostEqual(image[340, 348, 2].item(), 0.0, places=6)

            # overlap of red and blue gaussian
            self.assertAlmostEqual(image[200, 348, 0].item(), 0.07165284, places=6)
            self.assertAlmostEqual(image[200, 348, 1].item(), 0.0, places=6)
            self.assertAlmostEqual(image[200, 348, 2].item(), 0.22922967, places=6)

            image = image.clip(0, 1).detach().cpu().numpy()
            cv2.imwrite("test_splat_cpu.png", (image * 255).astype(np.uint8)[..., ::-1])

    def test_splat_gpu(self):
        self.setup_gpu()

        with torch.no_grad():
            uv, gaussian_2d, culled_gaussians = project_and_cull(
                self.world_T_image, self.camera, self.gaussians
            )
            image = render_tiles_gpu(uv, culled_gaussians, gaussian_2d, self.camera)
            # near red gaussian center
            self.assertAlmostEqual(image[340, 348, 0].item(), 0.47702518, places=5)
            self.assertAlmostEqual(image[340, 348, 1].item(), 0.0, places=5)
            self.assertAlmostEqual(image[340, 348, 2].item(), 0.0, places=5)

            # overlap of red and blue gaussian
            self.assertAlmostEqual(image[200, 348, 0].item(), 0.07165284, places=5)
            self.assertAlmostEqual(image[200, 348, 1].item(), 0.0, places=5)
            self.assertAlmostEqual(image[200, 348, 2].item(), 0.22922967, places=5)
            image = image.clip(0, 1).detach().cpu().numpy()
            cv2.imwrite("test_splat_gpu.png", (image * 255).astype(np.uint8)[..., ::-1])


if __name__ == "__main__":
    unittest.main()
