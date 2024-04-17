import numpy as np
import torch
import unittest

from splat_py.depth import render_depth
from splat_py.utils import inverse_sigmoid_torch
from gaussian_test_data import get_test_data


class TestRenderDepth(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available())
        self.device = torch.device("cuda")
        self.gaussians, self.camera, self.camera_T_world = get_test_data(self.device)
        self.gaussians.opacity = inverse_sigmoid_torch(self.gaussians.opacity)

    def test_rasterize_no_sh(self):
        near_thresh = 0.3
        cull_mask_padding = 10
        mh_dist = 3.0

        alpha_threshold = 0.2
        depth_image = render_depth(
            self.gaussians,
            alpha_threshold,
            self.camera_T_world,
            self.camera,
            near_thresh,
            cull_mask_padding,
            mh_dist,
        )
        # near red gaussian center
        self.assertAlmostEqual(depth_image[340, 348].item(), 17.29551887512207, places=5)

        # overlap of red and blue gaussian, blue is in front of red
        self.assertAlmostEqual(depth_image[200, 348].item(), 13.205718040466309, places=5)
