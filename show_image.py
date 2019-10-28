#!/usr/bin/python
# Python code for RaspberryPi to communicate with
# Teensy driving the LED display
#
# See here:
# http://blog.oscarliang.net/connect-raspberry-pi-and-arduino-usb-cable/

import os
import logging
import time
from datetime import datetime
import numpy as np
from scipy import ndimage
import display1593 as display
import argparse

parser = argparse.ArgumentParser(description='Display image.')
parser.add_argument('filename', type=str,
                    help='Filename of image to display.')
args = parser.parse_args()

images_dir = 'images'

logging.basicConfig(
	filename='logfile.txt',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Image brightness is based on photo resistor readings
brightness_levels = {
    0: 4.5,
    1: 12.2323,
    2: 33.2508,
    3: 90.3849,
    4: 245.692,
    5: 667.859,
    6: 1815.43,
    7: 4934.85,
    8: 13414.3,
    9: 36463.9,
    10: 99119.1,
    11: 269434
}
# Convert to numpy array for easier indexing
brightness_levels = np.array(list(brightness_levels.items()))


def calculate_brightness_level(brightness):
    """Converts a reading from the photo-resistor (typically
    in the range 2.2 to 160.0) into an integer brightness
    level for use with the show_image_calibrated method of
    Display1593 instances.
    """
    if brightness > brightness_levels[:, 1].max():
        level = brightness_levels[:, 0].max()
    else:
        idx = np.argmax(brightness_levels[:, 1] > brightness)
        level = brightness_levels[idx, 0]
    return int(level)


def main():

    logging.info("\n\n------------------- show_image.py -------------------\n")

    dis = display.Display1593()
    dis.connect()
    dis.clear()

    # Parameter for calculating exponentially-weighted
    # moving average (EWMA)
    alpha = 0.1

    # Initialize with average of series of readings from sensor
    brightness_ewma = np.array(
        [dis.getBrightness() for i in range(5)]
    ).mean()
    brightness_level = 0  # Image will be displayed first time

    while True:

        # Get current time
        start_time = datetime.now()
        sec = start_time.second

        # Read light level from photo resistor
        brightness = dis.getBrightness()

        # Update EWMA
        brightness_ewma = alpha * brightness + (1 - alpha) * brightness_ewma

        # See if image needs adjusting
        new_level = calculate_brightness_level(brightness_ewma)
        if new_level != brightness_level:
            brightness_level = new_level
            logging.info("Brightness: %d", brightness_level)
            logging.info("Showing image %s", args.filename.__repr__())
            f = os.path.join(images_dir, args.filename)
            dis.show_image_calibrated(f, brightness=brightness_level)

        #logging.info("Brightness: {:.2f}, {}".format(
        #             brightness_ewma, new_level))
        while (datetime.now().second == sec):
            time.sleep(0.05)  # Short wait


if __name__ == '__main__':
    main()
