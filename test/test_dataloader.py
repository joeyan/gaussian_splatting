import unittest
import torch

from splat_py.constants import DATASET_PATH
from splat_py.dataloader import ColmapData


class TestColmapData(unittest.TestCase):
    """Test Colmap dataloader"""

    def setUp(self):
        self.colmap_directory_path = DATASET_PATH
        self.device = torch.device("cpu")
        self.colmap_data = ColmapData(self.colmap_directory_path, self.device, downsample_factor=8)

    def test_init(self):
        """Test Data Loading"""
        self.assertEqual(self.colmap_data.colmap_directory_path, self.colmap_directory_path)

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

        gaussians = self.colmap_data.create_gaussians()
        self.assertEqual(gaussians.xyz.shape[0], 138766)
        self.assertEqual(gaussians.xyz.shape[1], 3)
        self.assertEqual(gaussians.rgb.shape[0], 138766)
        self.assertEqual(gaussians.rgb.shape[1], 3)
        self.assertEqual(gaussians.opacity.shape[0], 138766)
        self.assertEqual(gaussians.opacity.shape[1], 1)
        self.assertEqual(gaussians.scale.shape[0], 138766)
        self.assertEqual(gaussians.scale.shape[1], 3)
        self.assertEqual(gaussians.quaternion.shape[0], 138766)
        self.assertEqual(gaussians.quaternion.shape[1], 4)

        self.assertAlmostEqual(gaussians.xyz[0, 0].item(), 5.048415184)
        self.assertAlmostEqual(gaussians.xyz[0, 1].item(), 1.673997640)
        self.assertAlmostEqual(gaussians.xyz[0, 2].item(), -1.014126658)

        self.assertAlmostEqual(gaussians.rgb[0, 0].item(), 0.27803197503089905)
        self.assertAlmostEqual(gaussians.rgb[0, 1].item(), 0.48655596375465393)
        self.assertAlmostEqual(gaussians.rgb[0, 2].item(), 0.06950799375772476)

        self.assertAlmostEqual(gaussians.opacity[0, 0].item(), -1.3862943649)

        self.assertAlmostEqual(gaussians.scale[0, 0].item(), -3.722839117050171)
        self.assertAlmostEqual(gaussians.scale[0, 1].item(), -3.722839117050171)
        self.assertAlmostEqual(gaussians.scale[0, 2].item(), -3.722839117050171)

        self.assertAlmostEqual(gaussians.quaternion[0, 0].item(), 1.0)
        self.assertAlmostEqual(gaussians.quaternion[0, 1].item(), 0.0)
        self.assertAlmostEqual(gaussians.quaternion[0, 2].item(), 0.0)
        self.assertAlmostEqual(gaussians.quaternion[0, 3].item(), 0.0)

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

        self.assertTrue(torch.allclose(images[0].world_T_image, expected_world_T_image, atol=1e-4))

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
