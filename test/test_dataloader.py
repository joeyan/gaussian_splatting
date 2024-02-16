import unittest
import sys
import torch

sys.path.append("../")
from dataloader import ColmapData
from options import GaussianSplattingOptions


# To download test data:
# wget http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
COLMAP_TEST_PATH = "/home/joe/Downloads/garden"


class TestColmapData(unittest.TestCase):
    """Test Colmap dataloader"""

    def setUp(self):
        self.colmap_directory_path = COLMAP_TEST_PATH
        self.device = torch.device("cpu")
        self.colmap_data = ColmapData(
            self.colmap_directory_path, self.device, downsample_factor=8
        )

    def test_init(self):
        """Test Data Loading"""
        self.assertEqual(
            self.colmap_data.colmap_directory_path, self.colmap_directory_path
        )

        # values for garden dataset
        self.assertEqual(len(self.colmap_data.image_info), 185)
        self.assertEqual(len(self.colmap_data.xyz), 138766)
        self.assertEqual(len(self.colmap_data.rgb), 138766)
        self.assertEqual(len(self.colmap_data.cameras), 1)

        # test image data
        self.assertEqual(self.colmap_data.image_info[1].id, 1)
        self.assertEqual(self.colmap_data.image_info[1].camera_id, 1)
        self.assertEqual(self.colmap_data.image_info[1].name, "DSC07956.JPG")
        self.assertEqual(len(self.colmap_data.image_info[1].point3D_ids), 11193)

    def test_create_gaussians(self):
        """Test Gaussian Creation from colmap dataset"""
        options = GaussianSplattingOptions()
        options.initial_opacity_value = 0.2
        options.mean_neighbor_dist_to_initial_scale_factor = 0.2

        gaussians = self.colmap_data.create_gaussians(options)
        self.assertEqual(gaussians.xyz.shape[0], 138766)
        self.assertEqual(gaussians.xyz.shape[1], 3)
        self.assertEqual(gaussians.rgb.shape[0], 138766)
        self.assertEqual(gaussians.rgb.shape[1], 3)
        self.assertEqual(gaussians.opacities.shape[0], 138766)
        self.assertEqual(gaussians.opacities.shape[1], 1)
        self.assertEqual(gaussians.scales.shape[0], 138766)
        self.assertEqual(gaussians.scales.shape[1], 3)
        self.assertEqual(gaussians.quaternions.shape[0], 138766)
        self.assertEqual(gaussians.quaternions.shape[1], 4)

        self.assertAlmostEqual(gaussians.xyz[0, 0].item(), 5.048415184)
        self.assertAlmostEqual(gaussians.xyz[0, 1].item(), 1.673997640)
        self.assertAlmostEqual(gaussians.xyz[0, 2].item(), -1.014126658)

        self.assertAlmostEqual(gaussians.rgb[0, 0].item(), 0.078431375)
        self.assertAlmostEqual(gaussians.rgb[0, 1].item(), 0.137254908)
        self.assertAlmostEqual(gaussians.rgb[0, 2].item(), 0.019607843)

        # self.assertAlmostEqual(gaussians.opacities[0, 0].item(), -1.3862943649)

        self.assertAlmostEqual(gaussians.scales[0, 0].item(), -5.10913324356)
        self.assertAlmostEqual(gaussians.scales[0, 1].item(), -5.10913324356)
        self.assertAlmostEqual(gaussians.scales[0, 2].item(), -5.10913324356)

        self.assertAlmostEqual(gaussians.quaternions[0, 0].item(), 1.0)
        self.assertAlmostEqual(gaussians.quaternions[0, 1].item(), 0.0)
        self.assertAlmostEqual(gaussians.quaternions[0, 2].item(), 0.0)
        self.assertAlmostEqual(gaussians.quaternions[0, 3].item(), 0.0)

    def test_load_capture_info(self):
        """Test loading Images, Cameras"""
        images = self.colmap_data.get_images()
        self.assertEqual(len(images), 185)
        self.assertEqual(images[0].image.shape[0], 420)
        self.assertEqual(images[0].image.shape[1], 648)
        self.assertEqual(images[0].camera_id, 1)
        self.assertEqual(images[0].world_T_image.shape[0], 4)
        self.assertEqual(images[0].world_T_image.shape[1], 4)

        expected_world_T_image = torch.tensor(
            [
                [0.9999, 0.0089, 0.0073, -0.3283],
                [-0.0106, 0.9568, 0.2905, -1.9260],
                [-0.0044, -0.2906, 0.9568, 3.9581],
                [0.0000, 0.0000, 0.0000, 1.0000],
            ],
            dtype=torch.float32,
        )

        self.assertTrue(
            torch.allclose(images[0].world_T_image, expected_world_T_image, atol=1e-4)
        )

        cameras = self.colmap_data.get_cameras()
        self.assertEqual(len(cameras), 1)

        # using 8x downsample factor
        expected_K = torch.tensor(
            [[480.6123, 0.0, 324.1875], [0.0, 481.5445, 210.0625], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(cameras[1].K, expected_K, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
