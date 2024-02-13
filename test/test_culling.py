import math
import unittest
import sys
import torch
import cv2

sys.path.append("../")
from splat import compute_obb, compute_bbox_tile_intersection
from structs import Tiles


def plot_bbox(bbox, image, color=(0, 255, 0), thickness=1):
    bbox = bbox.cpu().numpy().astype(int)
    cv2.line(image, bbox[0, :], bbox[1, :], color, thickness)  # top left to top right
    cv2.line(
        image, bbox[1, :], bbox[3, :], color, thickness
    )  # top right to bottom right
    cv2.line(
        image, bbox[3, :], bbox[2, :], color, thickness
    )  # bottom right to bottom left
    cv2.line(image, bbox[2, :], bbox[0, :], color, thickness)  # bottom left to top left


def save_bboxes_to_image(bboxes, filename="culling_test_0.png"):
    image = torch.zeros((1024, 1024, 3), dtype=torch.uint8).numpy()
    for bbox in bboxes:
        # scale to avoid pixelation
        bbox = bbox * 20
        plot_bbox(bbox, image)
    cv2.imwrite(filename, image)


def draw_intersect_mask(intersect_mask, tiles, bbox, filename="culling_test_1.png"):
    image = torch.zeros(
        (tiles.image_height, tiles.image_width, 3), dtype=torch.uint8
    ).numpy()
    for tile in range(tiles.tile_count):
        if intersect_mask[tile]:
            cv2.rectangle(
                image,
                tiles.tile_corners[tile, 0, :].cpu().numpy(),
                tiles.tile_corners[tile, 3, :].cpu().numpy(),
                color=(255, 255, 255),
                thickness=-1,
            )
    plot_bbox(bbox, image)
    cv2.imwrite(filename, image)


class TestCulling(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")

    def test_obb_aligned(self):
        uv = torch.tensor(
            [
                15.0,
                15.0,
            ],
            dtype=torch.float32,
            device=self.device,
        )
        sigma_image = torch.tensor(
            [
                9.0,
                5.0,
                5.0,
                4.0,
            ],
            dtype=torch.float32,
            device=self.device,
        ).reshape(2, 2)
        obb = compute_obb(uv, sigma_image, mh_dist=1.0)

        self.assertAlmostEqual(obb[0, 0].item(), 12.5437, places=4)
        self.assertAlmostEqual(obb[0, 1].item(), 12.3606, places=4)
        self.assertAlmostEqual(obb[1, 0].item(), 18.4593, places=4)
        self.assertAlmostEqual(obb[1, 1].item(), 16.0166, places=4)
        self.assertAlmostEqual(obb[2, 0].item(), 11.5407, places=4)
        self.assertAlmostEqual(obb[2, 1].item(), 13.9834, places=4)
        self.assertAlmostEqual(obb[3, 0].item(), 17.4563, places=4)
        self.assertAlmostEqual(obb[3, 1].item(), 17.6394, places=4)

    def test_obb_intersection_0(self):
        tile = torch.tensor(
            [
                0.0,
                0.0,  # top left
                16.0,
                0.0,  # top right
                0.0,
                16.0,  # bottom left
                16.0,
                16.0,  # bottom right
            ],
            dtype=torch.float32,
            device=self.device,
        ).reshape(4, 2)

        # square centered at zero, rotated ccw by 45 degrees
        obb = torch.tensor(
            [
                -5.0,
                0.0,  # top left
                0.0,
                -5.0,  # top right
                0.0,
                5.0,  # bottom left
                5.0,
                0.0,  # bottom right
            ],
            dtype=torch.float32,
            device=self.device,
        ).reshape(4, 2)
        self.assertTrue(compute_bbox_tile_intersection(obb, tile))

        # tiny intersection
        offset = torch.tensor(
            [
                16.0 + 2.5 - 0.01,
                16.0 + 2.5 - 0.01,
            ],
            dtype=torch.float32,
            device=self.device,
        )
        obb = torch.add(obb, offset)
        self.assertTrue(compute_bbox_tile_intersection(obb, tile))

        # no intersection
        offset = torch.tensor(
            [
                0.02,
                0.02,
            ],
            dtype=torch.float32,
            device=self.device,
        )
        obb = torch.add(obb, offset)
        self.assertFalse(compute_bbox_tile_intersection(obb, tile))

    def test_obb_intersection_1(self):
        uv = torch.tensor(
            [
                50.0,
                50.0,
            ],
            dtype=torch.float32,
            device=self.device,
        )
        sigma_image = torch.tensor(
            [
                900.0,
                500.0,
                500.0,
                400.0,
            ],
            dtype=torch.float32,
            device=self.device,
        ).reshape(2, 2)
        obb = compute_obb(uv, sigma_image)

        tiles = Tiles(128, 128, self.device)

        intersecting_tiles = []
        intersect_mask = torch.zeros(
            tiles.tile_count, dtype=torch.bool, device=self.device
        )
        for tile in range(tiles.tile_count):
            intersect_mask[tile] = compute_bbox_tile_intersection(
                obb, tiles.tile_corners[tile].float()
            )
            if intersect_mask[tile]:
                intersecting_tiles.append(tile)

        expected_intersecting_tiles = {
            35,
            36,
            37,
            9,
            10,
            16,
            17,
            18,
            19,
            20,
            25,
            26,
            27,
            28,
            29,
        }
        self.assertEqual(set(intersecting_tiles), expected_intersecting_tiles)

    def test_obb_intersection_2(self):
        uv = torch.tensor(
            [
                632.8523,
                248.8553,
            ],
            dtype=torch.float32,
            device=self.device,
        )
        sigma_image = torch.tensor(
            [
                1.3287e04,
                9.7362e03,
                9.7362e03,
                7.3605e03,
            ],
            dtype=torch.float32,
            device=self.device,
        ).reshape(2, 2)
        obb = compute_obb(uv, sigma_image, mh_dist=3.0)

        tiles = Tiles(480, 640, self.device)

        intersecting_tiles = []
        intersect_mask = torch.zeros(
            tiles.tile_count, dtype=torch.bool, device=self.device
        )
        for tile in range(tiles.tile_count):
            intersect_mask[tile] = compute_bbox_tile_intersection(
                obb, tiles.tile_corners[tile].float()
            )
            if intersect_mask[tile]:
                intersecting_tiles.append(tile)

        draw_intersect_mask(intersect_mask, tiles, obb)


if __name__ == "__main__":
    unittest.main()
