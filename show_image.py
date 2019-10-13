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

# Programmed hourly brightness levels (0-8)
bcycle = {
    0: 1,
    1: 1,
    2: 1,
    3: 1,
    4: 1,
    5: 1,
    6: 1,
    7: 2,
    8: 2,
    9: 3,
    10: 4,
    11: 4,
    12: 4,
    13: 4,
    14: 4,
    15: 4,
    16: 4,
    17: 4,
    18: 3,
    19: 2,
    20: 2,
    21: 1,
    22: 1,
    23: 1
}


def main():

    logging.info("\n\n------------------- show_image.py -------------------\n")

    dis = display.Display1593()
    dis.connect()

    dis.clear()

    # Get current time
    start_time = datetime.now()
    hr, mn, sc = (start_time.hour, start_time.minute, start_time.second)

    brightness = None

    while True:

        # Get current time
        start_time = datetime.now()
        hr = start_time.hour

        # Set brightness level
        if brightness != bcycle[hr % 24]:
            brightness = bcycle[hr % 24]
            logging.info("Brightness: %d", brightness)

            logging.info("Showing image %s", args.filename.__repr__())
            f = os.path.join(images_dir, args.filename)
            dis.show_image_calibrated(f, brightness=brightness)

        logging.info("Waiting...")
        while (datetime.now().hour == hr):
            time.sleep(10)


if __name__ == '__main__':
    main()
