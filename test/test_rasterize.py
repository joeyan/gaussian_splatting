import cv2
import numpy as np
import torch
import unittest

from splat_py.rasterize import rasterize
from splat_py.utils import inverse_sigmoid_torch

from gaussian_test_data import get_test_data

SAVE_DEBUG = False


class TestRasterize(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available())
        self.device = torch.device("cuda")
        self.gaussians, self.camera, self.camera_T_world = get_test_data(self.device)
        self.gaussians.opacity = inverse_sigmoid_torch(self.gaussians.opacity)

    def test_rasterize_no_sh(self):
        near_thresh = 0.3
        cull_mask_padding = 10
        mh_dist = 3.0
        use_sh_precompute = True

        background_rgb = torch.zeros(3, device=self.device, dtype=self.gaussians.rgb.dtype)
        image, _, _ = rasterize(
            self.gaussians,
            self.camera_T_world,
            self.camera,
            near_thresh,
            cull_mask_padding,
            mh_dist,
            use_sh_precompute,
            background_rgb,
        )
        if SAVE_DEBUG:
            debug_image = image.clip(0, 1).detach().cpu().numpy()
            cv2.imwrite(
                "/tmp/test_rasterize_no_sh.png", (debug_image * 255).astype(np.uint8)[..., ::-1]
            )

        # near red gaussian center
        self.assertAlmostEqual(image[340, 348, 0].item(), 0.47698545455932617, places=5)
        self.assertAlmostEqual(image[340, 348, 1].item(), 0.0, places=5)
        self.assertAlmostEqual(image[340, 348, 2].item(), 0.0, places=5)

        # overlap of red and blue gaussian, blue is in front of red
        self.assertAlmostEqual(image[200, 348, 0].item(), 0.03330837935209274, places=5)
        self.assertAlmostEqual(image[200, 348, 1].item(), 0.0, places=5)
        self.assertAlmostEqual(image[200, 348, 2].item(), 0.267561137676239, places=5)

    def test_rasterize_full_sh_use_precompute(self):
        near_thresh = 0.3
        cull_mask_padding = 10
        mh_dist = 3.0
        use_sh_precompute = True
        self.gaussians.sh = (
            torch.ones((self.gaussians.xyz.shape[0], 3, 15), device=self.device) * 0.1
        )
        background_rgb = torch.zeros(3, device=self.device, dtype=self.gaussians.rgb.dtype)
        image, _, _ = rasterize(
            self.gaussians,
            self.camera_T_world,
            self.camera,
            near_thresh,
            cull_mask_padding,
            mh_dist,
            use_sh_precompute,
            background_rgb,
        )
        if SAVE_DEBUG:
            debug_image = image.clip(0, 1).detach().cpu().numpy()
            cv2.imwrite(
                "/tmp/test_rasterize_full_sh_use_precompute.png",
                (debug_image * 255).astype(np.uint8)[..., ::-1],
            )

        # near red gaussian center
        self.assertAlmostEqual(image[340, 348, 0].item(), 0.5362688899040222, places=5)
        self.assertAlmostEqual(image[340, 348, 1].item(), 0.05928343906998634, places=5)
        self.assertAlmostEqual(image[340, 348, 2].item(), 0.05928343906998634, places=5)

        # overlap of red and blue gaussian, blue is in front of red
        self.assertAlmostEqual(image[200, 348, 0].item(), 0.10543855279684067, places=5)
        self.assertAlmostEqual(image[200, 348, 1].item(), 0.07212823629379272, places=5)
        self.assertAlmostEqual(image[200, 348, 2].item(), 0.3396894335746765, places=5)

    def test_rasterize_full_sh_use_per_pixel_viewdir(self):
        near_thresh = 0.3
        cull_mask_padding = 10
        mh_dist = 3.0
        use_sh_precompute = False
        self.gaussians.sh = (
            torch.ones((self.gaussians.xyz.shape[0], 3, 15), device=self.device) * 0.1
        )

        background_rgb = torch.zeros(3, device=self.device, dtype=self.gaussians.rgb.dtype)
        image, _, _ = rasterize(
            self.gaussians,
            self.camera_T_world,
            self.camera,
            near_thresh,
            cull_mask_padding,
            mh_dist,
            use_sh_precompute,
            background_rgb,
        )
        if SAVE_DEBUG:
            debug_image = image.clip(0, 1).detach().cpu().numpy()
            cv2.imwrite(
                "/tmp/test_rasterize_full_sh_use_per_pixel_viewdir.png",
                (debug_image * 255).astype(np.uint8)[..., ::-1],
            )

        # near red gaussian center
        self.assertAlmostEqual(image[340, 348, 0].item(), 0.5328576564788818, places=5)
        self.assertAlmostEqual(image[340, 348, 1].item(), 0.05587226152420044, places=5)
        self.assertAlmostEqual(image[340, 348, 2].item(), 0.05587226152420044, places=5)

        # overlap of red and blue gaussian, blue is in front of red
        self.assertAlmostEqual(image[200, 348, 0].item(), 0.06694115698337555, places=5)
        self.assertAlmostEqual(image[200, 348, 1].item(), 0.033630844205617905, places=5)
        self.assertAlmostEqual(image[200, 348, 2].item(), 0.30119192600250244, places=5)


if __name__ == "__main__":
    unittest.main()
