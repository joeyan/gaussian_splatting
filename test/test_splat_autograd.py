import unittest
import cv2
import numpy as np
import torch
from cuda_autograd_functions import RenderImage
from tile_culling import match_gaussians_to_tiles_gpu
from structs import Tiles


class TestSplatAutograd(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available())
        self.device = torch.device("cuda")

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
        self.sigma_image = torch.tensor(
            [
                [[1.3287e03, 9.7362e02], [9.7362e02, 7.3605e02]],
                [[90.0, 20.0], [20.0, 60.0]],
                [[776.215, -2464.463], [-2464.463, 8276.755]],
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
            self.uv.float(), self.tiles, self.sigma_image.float(), mh_dist=3.0
        )

        self.opacities = torch.ones(
            self.uv.shape[0],
            1,
            dtype=torch.float64,
            device=self.device,
            requires_grad=True,
        )

        self.rgb = (
            torch.ones(
                self.uv.shape[0],
                3,
                dtype=torch.float64,
                device=self.device,
                requires_grad=True,
            )
            * 0.5
        )
        self.rgb[0, 0] = 0.0
        self.rgb[1, 1] = 0.0

    def test_render_image_grad(self):
        image_size = torch.tensor([40, 60], dtype=torch.int, device=self.device)

        image = RenderImage.apply(
            self.rgb,
            self.opacities,
            self.uv,
            self.sigma_image,
            self.splat_start_end_idx_by_tile_idx,
            self.gaussian_indices_per_tile,
            image_size,
        )

        image = image.clip(0, 1).detach().cpu().numpy()
        image = (image * 255).astype(np.uint8)[..., ::-1]
        cv2.imwrite("grad_render.png", image)

        test = torch.autograd.gradcheck(
            RenderImage.apply,
            (
                self.rgb,
                self.opacities,
                self.uv,
                self.sigma_image,
                self.splat_start_end_idx_by_tile_idx,
                self.gaussian_indices_per_tile,
                image_size,
            ),
            raise_exception=True,
        )
        print(test)


if __name__ == "__main__":
    unittest.main()
