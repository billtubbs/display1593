import unittest

from display1593.data.ledArray_data_1593 import build_lookup_table


class BuildLookupTableTests(unittest.TestCase):
    def test_toy_layout(self):
        # 2 controllers x 2 strips, ragged strip lengths, addresses
        # spaced out by a max_leds_per_strip larger than any strip
        leds_per_strip = ((2, 3), (1, 2))
        max_leds_per_strip = 10
        number_of_controllers = 2
        number_of_strips = 2

        result = build_lookup_table(
            leds_per_strip,
            max_leds_per_strip,
            number_of_controllers,
            number_of_strips,
        )

        expected = [
            (0, 0),
            (0, 1),  # controller 0, strip 0
            (0, 10),
            (0, 11),
            (0, 12),  # controller 0, strip 1
            (1, 0),  # controller 1, strip 0
            (1, 10),
            (1, 11),  # controller 1, strip 1
        ]
        self.assertEqual(result, expected)

    def test_result_length_matches_total_led_count(self):
        leds_per_strip = ((2, 3), (1, 2))

        result = build_lookup_table(leds_per_strip, 10, 2, 2)

        self.assertEqual(
            len(result), sum(n for strip in leds_per_strip for n in strip)
        )

    def test_single_controller_single_strip(self):
        result = build_lookup_table(((3,),), 10, 1, 1)

        self.assertEqual(result, [(0, 0), (0, 1), (0, 2)])


if __name__ == "__main__":
    unittest.main()
