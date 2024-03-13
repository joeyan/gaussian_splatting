import cv2
import numpy as np
import torch
import unittest

from splat_py.splat import splat
from splat_py.utils import inverse_sigmoid_torch

from gaussian_test_data import get_test_data


class TestSplatFull(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available())
        self.device = torch.device("cuda")
        self.gaussians, self.camera, self.world_T_image = get_test_data(self.device)
        self.gaussians.opacities = inverse_sigmoid_torch(self.gaussians.opacities)
        self.gaussians.rgb = inverse_sigmoid_torch(self.gaussians.rgb)

    def test_splat_gpu(self):
        image, _ = splat(self.gaussians, self.world_T_image, self.camera)

        # near red gaussian center
        self.assertAlmostEqual(image[340, 348, 0].item(), 0.476977467, places=5)
        self.assertAlmostEqual(image[340, 348, 1].item(), 9.539548773e-05, places=5)
        self.assertAlmostEqual(image[340, 348, 2].item(), 9.539548773e-05, places=5)

        # overlap of red and blue gaussian, blue is in front of red
        self.assertAlmostEqual(image[200, 348, 0].item(), 0.033361889, places=5)
        self.assertAlmostEqual(image[200, 348, 1].item(), 6.01712417e-05, places=5)
        self.assertAlmostEqual(image[200, 348, 2].item(), 0.26755452, places=5)
        image = image.clip(0, 1).detach().cpu().numpy()
        cv2.imwrite("test_splat.png", (image * 255).astype(np.uint8)[..., ::-1])


if __name__ == "__main__":
    unittest.main()
