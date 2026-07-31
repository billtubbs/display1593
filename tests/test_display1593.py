import unittest

import numpy as np

from display1593.display1593 import Display1593
from data.ledArray_data_1593 import num_cells


class NearestNeighboursAttributeTests(unittest.TestCase):
    def setUp(self):
        self.display = Display1593()

    def test_same_shape(self):
        self.assertEqual(
            self.display.nearest_neighbours.shape,
            self.display.nearest_neighbour_distances.shape,
        )

    def test_row_count_matches_expected_number_of_leds(self):
        self.assertEqual(self.display.nearest_neighbours.shape[0], num_cells)
        self.assertEqual(
            self.display.nearest_neighbour_distances.shape[0], num_cells
        )

    def test_nearest_neighbours_dtype_is_uint16(self):
        self.assertEqual(self.display.nearest_neighbours.dtype, np.uint16)


if __name__ == "__main__":
    unittest.main()
