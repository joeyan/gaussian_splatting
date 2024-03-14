import unittest
import torch

from splat_py.structs import Tiles


class TestTiles(unittest.TestCase):
    """Test tile generation"""

    def test_initialization(self):
        image_height = 1080
        image_width = 1920
        device = torch.device("cpu")

        tiles = Tiles(image_height, image_width, device)

        self.assertEqual(tiles.image_height, image_height)
        self.assertEqual(tiles.image_width, image_width)
        self.assertEqual(tiles.device, device)

        self.assertEqual(tiles.image_height_padded, 1088)
        self.assertEqual(tiles.image_width_padded, 1920)

        self.assertEqual(tiles.y_tiles_count, 68)
        self.assertEqual(tiles.x_tiles_count, 120)
        self.assertEqual(tiles.tile_count, 8160)


if __name__ == "__main__":
    unittest.main()
