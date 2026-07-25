from itertools import cycle
from collections import deque
import numpy as np
import serial
from serial_comm import (
    connect_to_arduino,
    send_data_to_arduino,
    receive_data_from_arduino,
)
from display1593 import *
import os
import time


def main():
    logger.info("=" * 35)
    filename = os.path.basename(__file__)
    logger.info(f"{filename} started.")
    with Display1593() as dis:
        dis.set_led(0, RED)

        for _ in range(1):
            # Test clear_all method
            dis.clear_all()

            # Test show_now method
            dis.show_now()

            # Test set_led method
            dis.clear_all()
            dis.set_led(0, RED)
            dis.set_led(797, BLUE)
            dis.set_led(798, RED)
            dis.set_led(1592, BLUE)
            dis.show_now()

    #         # Test set_all_leds method
    #         rgb_array = np.stack([
    #             (i % 32, i % 32, i % 32) for i in range(1593)
    #         ]).astype('uint8')
    #         assert rgb_array.shape == (1593, 3)
    #         dis.set_all_leds(rgb_array)
    #         dis.show_now()
    #
    #         # Test set_leds_one_colour method
    #         dis.clear_all()
    #         rgb = [12, 4, 36]
    #         led_range = 600, 1000
    #         leds = np.arange(*led_range).astype('int32')
    #         dis.set_leds_one_colour(leds, rgb)
    #         dis.show_now()
    #
    #         # Test set_all_leds_one_colour method
    #         dis.clear_all()
    #         rgb = [36, 4, 12]
    #         dis.set_all_leds_one_colour(rgb)
    #         dis.show_now()
    #
    #         # Test set_leds method
    #         dis.clear_all()
    #         start_led = 700
    #         colors = [BLACK] + [RED, GREEN, BLUE] * 20
    #         rgb_array = np.stack(colors).astype('uint8')
    #         t_last = time.time()
    #         for iter in range(200):
    #             leds = [(start_led + i) % 1593 for i in range(len(colors))]
    #             dis.set_leds(leds, rgb_array)
    #             t_now = time.time()
    #             t_next = t_last + 0.05
    #             logger.info(f"Wait {t_next - t_now:.3f}s.")
    #             while t_now < t_next:
    #                 t_now = time.time()
    #             dis.show_now()
    #             t_last = t_now
    #             start_led = (start_led + 1) % 1593

    logger.info(f"{filename} ended.")


if __name__ == "__main__":
    main()
