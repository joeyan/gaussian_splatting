import unittest
import torch

from splat_py.cuda_autograd_functions import RenderImage
from splat_py.structs import Tiles, Camera
from splat_py.tile_culling import match_gaussians_to_tiles_gpu
from splat_py.utils import compute_rays_in_world_frame


class TestRasterizeAutograd(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available())
        self.device = torch.device("cuda")

        K = torch.tensor(
            [
                [43.0, 0.0, 30.0],
                [0.0, 41.0, 20.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
            device=self.device,
        )
        self.camera = Camera(60, 40, K)
        self.camera_T_world = torch.eye(4, dtype=torch.float64, device=self.device)
        self.rays = compute_rays_in_world_frame(self.camera, self.camera_T_world)

        self.uv = torch.tensor(
            [
                [32.8523, 24.88553],
                [25.0, 25.0],
                [45.339926, 13.85983],
            ],
            dtype=torch.float64,
            device=self.device,
            requires_grad=True,
        )
        self.conic = torch.tensor(
            [
                [1.3287e03, 9.7362e02 * 2, 7.3605e02],
                [90.0, 20.0 * 2, 60.0],
                [776.215, -2464.463 * 2, 8276.755],
            ],
            dtype=torch.float64,
            device=self.device,
            requires_grad=True,
        )
        self.tiles = Tiles(40, 60, self.device)

        (
            self.gaussian_indices_per_tile,
            self.splat_start_end_idx_by_tile_idx,
            self.tile_idx_by_splat_idx,
        ) = match_gaussians_to_tiles_gpu(
            self.uv.float(), self.tiles, self.conic.float(), mh_dist=3.0
        )
        self.opacity = torch.ones(
            self.uv.shape[0],
            1,
            dtype=torch.float64,
            device=self.device,
            requires_grad=True,
        )

    def test_rasterize_image_grad_SH_0(self):
        image_size = torch.tensor([40, 60], dtype=torch.int, device=self.device)
        rgb = (
            torch.ones(
                self.uv.shape[0],
                3,
                dtype=torch.float64,
                device=self.device,
                requires_grad=True,
            )
            * 0.5
        )
        rgb[0, 0] = 0.0
        rgb[1, 1] = 0.0

        background_rgb = (
            torch.ones(3, dtype=torch.float64, device=self.device, requires_grad=False) * 0.5
        )
        test = torch.autograd.gradcheck(
            RenderImage.apply,
            (
                rgb,
                self.opacity,
                self.uv,
                self.conic,
                self.rays,
                self.splat_start_end_idx_by_tile_idx,
                self.gaussian_indices_per_tile,
                image_size,
                background_rgb,
            ),
            raise_exception=True,
        )
        self.assertTrue(test)

    def test_rasterize_image_grad_SH_0_no_background(self):
        image_size = torch.tensor([40, 60], dtype=torch.int, device=self.device)
        rgb = (
            torch.ones(
                self.uv.shape[0],
                3,
                dtype=torch.float64,
                device=self.device,
                requires_grad=True,
            )
            * 0.5
        )
        rgb[0, 0] = 0.0
        rgb[1, 1] = 0.0

        background_rgb = torch.zeros(
            3, dtype=torch.float64, device=self.device, requires_grad=False
        )
        test = torch.autograd.gradcheck(
            RenderImage.apply,
            (
                rgb,
                self.opacity,
                self.uv,
                self.conic,
                self.rays,
                self.splat_start_end_idx_by_tile_idx,
                self.gaussian_indices_per_tile,
                image_size,
                background_rgb,
            ),
            raise_exception=True,
        )
        self.assertTrue(test)

    def test_rasterize_image_grad_SH_4(self):
        test_size = torch.tensor([40, 60], dtype=torch.int, device=self.device)
        sh_coeff_4 = (
            torch.ones(
                self.uv.shape[0],
                3,
                4,
                dtype=torch.float64,
                device=self.device,
                requires_grad=True,
            )
            * 0.5
        )
        background_rgb = (
            torch.ones(3, dtype=torch.float64, device=self.device, requires_grad=False) * 0.5
        )
        test = torch.autograd.gradcheck(
            RenderImage.apply,
            (
                sh_coeff_4,
                self.opacity,
                self.uv,
                self.conic,
                self.rays,
                self.splat_start_end_idx_by_tile_idx,
                self.gaussian_indices_per_tile,
                test_size,
                background_rgb,
            ),
            raise_exception=True,
        )
        self.assertTrue(test)

    def test_rasterize_image_grad_SH_4_no_background(self):
        test_size = torch.tensor([40, 60], dtype=torch.int, device=self.device)
        sh_coeff_4 = (
            torch.ones(
                self.uv.shape[0],
                3,
                4,
                dtype=torch.float64,
                device=self.device,
                requires_grad=True,
            )
            * 0.5
        )
        background_rgb = torch.zeros(
            3, dtype=torch.float64, device=self.device, requires_grad=False
        )
        test = torch.autograd.gradcheck(
            RenderImage.apply,
            (
                sh_coeff_4,
                self.opacity,
                self.uv,
                self.conic,
                self.rays,
                self.splat_start_end_idx_by_tile_idx,
                self.gaussian_indices_per_tile,
                test_size,
                background_rgb,
            ),
            raise_exception=True,
        )
        self.assertTrue(test)

    def test_rasterize_image_grad_SH_9(self):
        test_size = torch.tensor([40, 60], dtype=torch.int, device=self.device)
        sh_coeff_9 = (
            torch.ones(
                self.uv.shape[0],
                3,
                9,
                dtype=torch.float64,
                device=self.device,
                requires_grad=True,
            )
            * 0.5
        )
        background_rgb = (
            torch.ones(3, dtype=torch.float64, device=self.device, requires_grad=False) * 0.5
        )
        test = torch.autograd.gradcheck(
            RenderImage.apply,
            (
                sh_coeff_9,
                self.opacity,
                self.uv,
                self.conic,
                self.rays,
                self.splat_start_end_idx_by_tile_idx,
                self.gaussian_indices_per_tile,
                test_size,
                background_rgb,
            ),
            raise_exception=True,
        )
        self.assertTrue(test)

    def test_rasterize_image_grad_SH_9_no_background(self):
        test_size = torch.tensor([40, 60], dtype=torch.int, device=self.device)
        sh_coeff_9 = (
            torch.ones(
                self.uv.shape[0],
                3,
                9,
                dtype=torch.float64,
                device=self.device,
                requires_grad=True,
            )
            * 0.5
        )
        background_rgb = torch.zeros(
            3, dtype=torch.float64, device=self.device, requires_grad=False
        )
        test = torch.autograd.gradcheck(
            RenderImage.apply,
            (
                sh_coeff_9,
                self.opacity,
                self.uv,
                self.conic,
                self.rays,
                self.splat_start_end_idx_by_tile_idx,
                self.gaussian_indices_per_tile,
                test_size,
                background_rgb,
            ),
            raise_exception=True,
        )
        self.assertTrue(test)

    def test_rasterize_image_grad_SH_16(self):
        test_size = torch.tensor([40, 60], dtype=torch.int, device=self.device)
        sh_coeff_16 = (
            torch.ones(
                self.uv.shape[0],
                3,
                16,
                dtype=torch.float64,
                device=self.device,
                requires_grad=True,
            )
            * 0.5
        )
        background_rgb = (
            torch.ones(3, dtype=torch.float64, device=self.device, requires_grad=False) * 0.5
        )
        test = torch.autograd.gradcheck(
            RenderImage.apply,
            (
                sh_coeff_16,
                self.opacity,
                self.uv,
                self.conic,
                self.rays,
                self.splat_start_end_idx_by_tile_idx,
                self.gaussian_indices_per_tile,
                test_size,
                background_rgb,
            ),
            raise_exception=True,
            atol=3e-5,
        )
        self.assertTrue(test)

    def test_rasterize_image_grad_SH_16_no_background(self):
        test_size = torch.tensor([40, 60], dtype=torch.int, device=self.device)
        sh_coeff_16 = (
            torch.ones(
                self.uv.shape[0],
                3,
                16,
                dtype=torch.float64,
                device=self.device,
                requires_grad=True,
            )
            * 0.5
        )
        background_rgb = torch.zeros(
            3, dtype=torch.float64, device=self.device, requires_grad=False
        )
        test = torch.autograd.gradcheck(
            RenderImage.apply,
            (
                sh_coeff_16,
                self.opacity,
                self.uv,
                self.conic,
                self.rays,
                self.splat_start_end_idx_by_tile_idx,
                self.gaussian_indices_per_tile,
                test_size,
                background_rgb,
            ),
            raise_exception=True,
            atol=3e-5,
        )
        self.assertTrue(test)


if __name__ == "__main__":
    unittest.main()
