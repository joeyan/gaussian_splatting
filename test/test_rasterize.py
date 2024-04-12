import cv2
import numpy as np
import torch
import unittest

from splat_py.rasterize import rasterize
from splat_py.utils import inverse_sigmoid_torch

from gaussian_test_data import get_test_data


class TestSplatFull(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available())
        self.device = torch.device("cuda")
        self.gaussians, self.camera, self.world_T_image = get_test_data(self.device)
        self.gaussians.opacities = inverse_sigmoid_torch(self.gaussians.opacities)

    def test_splat_gpu(self):
        image, _, _ = rasterize(self.gaussians, self.world_T_image, self.camera)
        debug_image = image.clip(0, 1).detach().cpu().numpy()
        cv2.imwrite("/tmp/test_splat.png", (debug_image * 255).astype(np.uint8)[..., ::-1])

        # near red gaussian center
        self.assertAlmostEqual(image[340, 348, 0].item(), 0.47698545455932617, places=5)
        self.assertAlmostEqual(image[340, 348, 1].item(), 0.0, places=5)
        self.assertAlmostEqual(image[340, 348, 2].item(), 0.0, places=5)

        # overlap of red and blue gaussian, blue is in front of red
        self.assertAlmostEqual(image[200, 348, 0].item(), 0.03330837935209274, places=5)
        self.assertAlmostEqual(image[200, 348, 1].item(), 0.0, places=5)
        self.assertAlmostEqual(image[200, 348, 2].item(), 0.267561137676239, places=5)


if __name__ == "__main__":
    unittest.main()
