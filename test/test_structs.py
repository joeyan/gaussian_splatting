import unittest
import sys

sys.path.append("../")
from structs import Tiles


class TestTiles(unittest.TestCase):
    """Test tile generation"""

    def test_initialization(self):
        image_height = 1080
        image_width = 1920
        device = "cpu"

        tiles = Tiles(image_height, image_width, device)

        self.assertEqual(tiles.image_height, image_height)
        self.assertEqual(tiles.image_width, image_width)
        self.assertEqual(tiles.device, device)

        self.assertEqual(tiles.image_height_padded, 1088)
        self.assertEqual(tiles.image_width_padded, 1920)

        self.assertEqual(tiles.y_tiles_count, 68)
        self.assertEqual(tiles.x_tiles_count, 120)
        self.assertEqual(tiles.tile_count, 8160)

        self.assertTrue((tiles.tile_corners.shape == (8160, 4, 2)))
        # tile 0, top left
        self.assertEqual(tiles.tile_corners[0, 0, 0], 0)  # x
        self.assertEqual(tiles.tile_corners[0, 0, 1], 0)  # y
        # tile 0, top right
        self.assertEqual(tiles.tile_corners[0, 1, 0], 16)  # x
        self.assertEqual(tiles.tile_corners[0, 1, 1], 0)  # y
        # tile 0, bottom left
        self.assertEqual(tiles.tile_corners[0, 2, 0], 0)  # x
        self.assertEqual(tiles.tile_corners[0, 2, 1], 16)  # y
        # tile 0, bottom right
        self.assertEqual(tiles.tile_corners[0, 3, 0], 16)  # x
        self.assertEqual(tiles.tile_corners[0, 3, 1], 16)  # y

        # tile 120, top left
        self.assertEqual(tiles.tile_corners[119, 0, 0], 1904)  # x
        self.assertEqual(tiles.tile_corners[119, 0, 1], 0)  # y
        # tile 120, top right
        self.assertEqual(tiles.tile_corners[119, 1, 0], 1920)  # x
        self.assertEqual(tiles.tile_corners[119, 1, 1], 0)  # y
        # tile 120, bottom left
        self.assertEqual(tiles.tile_corners[119, 2, 0], 1904)  # x
        self.assertEqual(tiles.tile_corners[119, 2, 1], 16)  # y
        # tile 120, bottom right
        self.assertEqual(tiles.tile_corners[119, 3, 0], 1920)  # x
        self.assertEqual(tiles.tile_corners[119, 3, 1], 16)  # y

        # tile 8159, top left
        self.assertEqual(tiles.tile_corners[8159, 0, 0], 1904)
        self.assertEqual(tiles.tile_corners[8159, 0, 1], 1072)
        # tile 8159, top right
        self.assertEqual(tiles.tile_corners[8159, 1, 0], 1920)
        self.assertEqual(tiles.tile_corners[8159, 1, 1], 1072)
        # tile 8159, bottom left
        self.assertEqual(tiles.tile_corners[8159, 2, 0], 1904)
        self.assertEqual(tiles.tile_corners[8159, 2, 1], 1088)
        # tile 8159, bottom right
        self.assertEqual(tiles.tile_corners[8159, 3, 0], 1920)
        self.assertEqual(tiles.tile_corners[8159, 3, 1], 1088)


if __name__ == "__main__":
    unittest.main()
