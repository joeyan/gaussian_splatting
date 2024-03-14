import unittest
import math
import torch
import cv2

from splat_py.tile_culling import (
    match_gaussians_to_tiles_gpu,
)
from splat_py.structs import Tiles


def get_tile_corners(tiles):
    # compute top left, top right, bottom left, bottom right of each tile [x, y]
    tile_corners = torch.zeros(
        tiles.tile_count, 4, 2, dtype=torch.int32, device=tiles.device
    )
    for row in range(tiles.y_tiles_count):
        for col in range(tiles.x_tiles_count):
            # top left
            tile_corners[row * tiles.x_tiles_count + col, 0, 0] = (
                col * tiles.tile_edge_size
            )
            tile_corners[row * tiles.x_tiles_count + col, 0, 1] = (
                row * tiles.tile_edge_size
            )
            # top right
            tile_corners[row * tiles.x_tiles_count + col, 1, 0] = (
                col + 1
            ) * tiles.tile_edge_size
            tile_corners[row * tiles.x_tiles_count + col, 1, 1] = (
                row * tiles.tile_edge_size
            )
            # bottom left
            tile_corners[row * tiles.x_tiles_count + col, 2, 0] = (
                col * tiles.tile_edge_size
            )
            tile_corners[row * tiles.x_tiles_count + col, 2, 1] = (
                row + 1
            ) * tiles.tile_edge_size
            # bottom right
            tile_corners[row * tiles.x_tiles_count + col, 3, 0] = (
                col + 1
            ) * tiles.tile_edge_size
            tile_corners[row * tiles.x_tiles_count + col, 3, 1] = (
                row + 1
            ) * tiles.tile_edge_size
    return tile_corners


def compute_obb(
    uv,
    sigma_image,
    mh_dist,
):
    """
    https://cookierobotics.com/007/

    Compute the oriented bounding box of a 2D gaussian at a specific confidence level
    """
    a = sigma_image[0, 0]
    b = sigma_image[0, 1]
    c = sigma_image[1, 0]
    d = sigma_image[1, 1]

    # compute the two radii of the 2d gaussian
    left = (a + d) / 2
    right = torch.sqrt(torch.square(a - d) / 4 + b * c)
    lambda_1 = left + right
    r1 = mh_dist * torch.sqrt(lambda_1)  # major axis
    r2 = mh_dist * torch.sqrt(left - right)  # minor axis

    # compute angle of major axis
    # theta is ccw from +x axis
    if abs(b) < 1e-16:
        if a >= d:
            theta = 0
        else:
            theta = math.pi / 2
    else:
        theta = math.atan2((lambda_1 - a), b)

    R = torch.tensor(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=torch.float32,
        device=uv.device,
    )

    obb = torch.zeros(4, 2, dtype=torch.float32, device=uv.device)
    obb[0, :] = uv + R @ torch.tensor(
        [-r1, -r2], dtype=torch.float32, device=uv.device
    )  # top left aabb corner
    obb[1, :] = uv + R @ torch.tensor(
        [r1, -r2], dtype=torch.float32, device=uv.device
    )  # top right aabb corner
    obb[2, :] = uv + R @ torch.tensor(
        [-r1, r2], dtype=torch.float32, device=uv.device
    )  # bottom left aabb corner
    obb[3, :] = uv + R @ torch.tensor(
        [r1, r2], dtype=torch.float32, device=uv.device
    )  # bottom right aabb corner

    return obb


def compute_bbox_tile_intersection(
    bbox_gaussian,
    bbox_tile,
):
    """
    compute the intersection of a bbox and a tile bbox

    Use the split axis theorem with simplifications:
    1) There are 4 axes to check for overlap since there are two sets of parallel edges in each rectangle
    2) Taking the normal of each edge is not required since the edges in a rectangle are always perpendicular
    3) Two axes to check are the x and y axes since the tile bbox is axis aligned

    bbox format:
    torch.tensor([
        top left,
        top right,
        bottom left,
        bottom right
    ])

    For more details: https://dyn4j.org/2010/01/sat/
    """

    # check x axis for overlap
    min_x_gaussian = torch.min(bbox_gaussian[:, 0])
    max_x_gaussian = torch.max(bbox_gaussian[:, 0])
    min_x_tile = bbox_tile[0, 0]
    max_x_tile = bbox_tile[1, 0]

    if min_x_gaussian > max_x_tile or max_x_gaussian < min_x_tile:
        return False

    # check y axis
    min_y_gaussian = torch.min(bbox_gaussian[:, 1])
    max_y_gaussian = torch.max(bbox_gaussian[:, 1])
    min_y_tile = bbox_tile[0, 1]
    max_y_tile = bbox_tile[2, 1]

    if min_y_gaussian > max_y_tile or max_y_gaussian < min_y_tile:
        return False

    # bbox_gaussian axis 0: top left to top right
    axis_0 = bbox_gaussian[1, :] - bbox_gaussian[0, :]
    # bbox_gaussian axis 1: top left to bottom left
    axis_1 = bbox_gaussian[2, :] - bbox_gaussian[0, :]

    for axis in [axis_0, axis_1]:
        projected_gaussian_0 = torch.dot(axis, bbox_gaussian[0, :])
        projected_tile_0 = torch.dot(axis, bbox_tile[0, :])

        min_projected_gaussian = projected_gaussian_0
        max_projected_gaussian = projected_gaussian_0
        min_projected_tile = projected_tile_0
        max_projected_tile = projected_tile_0

        for i in range(1, 4):
            projected_gaussian_pt = torch.dot(axis, bbox_gaussian[i, :])
            min_projected_gaussian = torch.min(
                min_projected_gaussian, projected_gaussian_pt
            )
            max_projected_gaussian = torch.max(
                max_projected_gaussian, projected_gaussian_pt
            )

            projected_tile_pt = torch.dot(axis, bbox_tile[i, :])
            min_projected_tile = torch.min(min_projected_tile, projected_tile_pt)
            max_projected_tile = torch.max(max_projected_tile, projected_tile_pt)

        if (
            min_projected_gaussian > max_projected_tile
            or max_projected_gaussian < min_projected_tile
        ):
            return False

    return True


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


def draw_intersect_mask(intersect_mask, tiles, bboxes, filename="culling_test_1.png"):
    image = torch.zeros(
        (tiles.image_height, tiles.image_width, 3), dtype=torch.uint8
    ).numpy()

    tile_corners = get_tile_corners(tiles)
    for tile in range(tiles.tile_count):
        if intersect_mask[tile]:
            cv2.rectangle(
                image,
                tile_corners[tile, 0, :].cpu().numpy(),
                tile_corners[tile, 3, :].cpu().numpy(),
                color=(255, 255, 255),
                thickness=-1,
            )
    for bbox in bboxes:
        plot_bbox(bbox, image)
    cv2.imwrite(filename, image)


class TestCulling(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")

    def test_tile_corners(self):
        image_height = 1080
        image_width = 1920
        device = torch.device("cpu")

        tiles = Tiles(image_height, image_width, device)

        tile_corners = get_tile_corners(tiles)
        self.assertTrue((tile_corners.shape == (8160, 4, 2)))
        # tile 0, top left
        self.assertEqual(tile_corners[0, 0, 0], 0)  # x
        self.assertEqual(tile_corners[0, 0, 1], 0)  # y
        # tile 0, top right
        self.assertEqual(tile_corners[0, 1, 0], 16)  # x
        self.assertEqual(tile_corners[0, 1, 1], 0)  # y
        # tile 0, bottom left
        self.assertEqual(tile_corners[0, 2, 0], 0)  # x
        self.assertEqual(tile_corners[0, 2, 1], 16)  # y
        # tile 0, bottom right
        self.assertEqual(tile_corners[0, 3, 0], 16)  # x
        self.assertEqual(tile_corners[0, 3, 1], 16)  # y

        # tile 120, top left
        self.assertEqual(tile_corners[119, 0, 0], 1904)  # x
        self.assertEqual(tile_corners[119, 0, 1], 0)  # y
        # tile 120, top right
        self.assertEqual(tile_corners[119, 1, 0], 1920)  # x
        self.assertEqual(tile_corners[119, 1, 1], 0)  # y
        # tile 120, bottom left
        self.assertEqual(tile_corners[119, 2, 0], 1904)  # x
        self.assertEqual(tile_corners[119, 2, 1], 16)  # y
        # tile 120, bottom right
        self.assertEqual(tile_corners[119, 3, 0], 1920)  # x
        self.assertEqual(tile_corners[119, 3, 1], 16)  # y

        # tile 8159, top left
        self.assertEqual(tile_corners[8159, 0, 0], 1904)
        self.assertEqual(tile_corners[8159, 0, 1], 1072)
        # tile 8159, top right
        self.assertEqual(tile_corners[8159, 1, 0], 1920)
        self.assertEqual(tile_corners[8159, 1, 1], 1072)
        # tile 8159, bottom left
        self.assertEqual(tile_corners[8159, 2, 0], 1904)
        self.assertEqual(tile_corners[8159, 2, 1], 1088)
        # tile 8159, bottom right
        self.assertEqual(tile_corners[8159, 3, 0], 1920)
        self.assertEqual(tile_corners[8159, 3, 1], 1088)

    def test_obb_aligned(self):
        uv = torch.tensor(
            [
                [15.0, 15.0],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        sigma_image = torch.tensor(
            [
                [9.0, 5.0],
                [5.0, 4.0],
            ],
            dtype=torch.float32,
            device=self.device,
        )
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
                [0.0, 0.0],  # top left
                [16.0, 0.0],  # top right
                [0.0, 16.0],  # bottom left
                [16.0, 16.0],  # bottom right
            ],
            dtype=torch.float32,
            device=self.device,
        )

        # square centered at zero, rotated ccw by 45 degrees
        obb = torch.tensor(
            [
                [-5.0, 0.0],  # top left
                [0.0, -5.0],  # top right
                [0.0, 5.0],  # bottom left
                [5.0, 0.0],  # bottom right
            ],
            dtype=torch.float32,
            device=self.device,
        )
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
                [900.0, 500.0],
                [500.0, 400.0],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        obb = compute_obb(uv, sigma_image, mh_dist=1.0)

        tiles = Tiles(128, 128, self.device)

        intersecting_tiles = []
        intersect_mask = torch.zeros(
            tiles.tile_count, dtype=torch.bool, device=self.device
        )

        tile_corners = get_tile_corners(tiles)
        for tile in range(tiles.tile_count):
            intersect_mask[tile] = compute_bbox_tile_intersection(
                obb, tile_corners[tile].float()
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
                [1.3287e04, 9.7362e03],
                [9.7362e03, 7.3605e03],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        obb = compute_obb(uv, sigma_image, mh_dist=3.0)

        tiles = Tiles(480, 640, self.device)

        intersecting_tiles = []
        intersect_mask = torch.zeros(
            tiles.tile_count, dtype=torch.bool, device=self.device
        )
        tile_corners = get_tile_corners(tiles)

        for tile in range(tiles.tile_count):
            intersect_mask[tile] = compute_bbox_tile_intersection(
                obb, tile_corners[tile].float()
            )
            if intersect_mask[tile]:
                intersecting_tiles.append(tile)

        draw_intersect_mask(
            intersect_mask, tiles, [obb], filename="test_obb_intersection_2.png"
        )

    def test_compute_tiles_cuda(self):
        cuda_device = torch.device("cuda")
        uv = torch.tensor(
            [
                [632.8523, 248.8553],
                [50.0, 50.0],
                [453.39926, 138.5983],
            ],
            dtype=torch.float32,
            device=cuda_device,
        )
        sigma_image = torch.tensor(
            [
                [[1.3287e04, 9.7362e03], [9.7362e03, 7.3605e03]],
                [[900.0, 500.0], [500.0, 400.0]],
                [[776.215, -2464.463], [-2464.463, 8276.755]],
            ],
            dtype=torch.float32,
            device=cuda_device,
        )

        obb_0 = compute_obb(uv[0], sigma_image[0], mh_dist=1.0)
        obb_1 = compute_obb(uv[1], sigma_image[1], mh_dist=1.0)
        obb_2 = compute_obb(uv[2], sigma_image[2], mh_dist=1.0)

        tiles = Tiles(480, 640, cuda_device)
        (
            _,
            splat_start_end_idx_by_tile_idx,
            _,
        ) = match_gaussians_to_tiles_gpu(uv, tiles, sigma_image, mh_dist=1.0)

        intersect_mask = torch.zeros(
            tiles.tile_count, dtype=torch.bool, device=cuda_device
        )
        for tile in range(tiles.tile_count):
            intersect_mask[tile] = (
                splat_start_end_idx_by_tile_idx[tile]
                != splat_start_end_idx_by_tile_idx[tile + 1]
            )

        draw_intersect_mask(
            intersect_mask,
            tiles,
            [obb_0, obb_1, obb_2],
            filename="test_compute_tiles_cuda.png",
        )


if __name__ == "__main__":
    unittest.main()
