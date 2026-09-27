#!/usr/bin/env python3
"""Minimal smoke test for the display driver on the Raspberry Pi.

Run this on the Pi connected to the Teensy boards. It flashes a few LEDs on
both boards, waits briefly, then clears them again.
"""

import time

import numpy as np

from display1593 import Display1593


def main():
    with Display1593() as dis:
        print("Connected to display")

        dis.clear_all()
        dis.show_now()

        # A few sample LEDs across the layout, including both boards.
        leds = np.array([0, 1, 10, 100, 500, 700, 1500], dtype=np.int32)
        rgb = np.array(
            [
                (255, 0, 0),
                (0, 255, 0),
                (0, 0, 255),
                (255, 255, 0),
                (255, 0, 255),
                (0, 255, 255),
                (255, 255, 255),
            ],
            dtype=np.uint8,
        )

        dis.set_leds(leds, rgb)
        dis.show_now()
        print("Pattern sent")

        time.sleep(2)

        dis.clear_all()
        dis.show_now()
        print("Cleared")


if __name__ == "__main__":
    main()
