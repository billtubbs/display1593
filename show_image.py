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

# Programmed hourly brightness levels
bcycle = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 0,
    8: 1,
    9: 4,
    10: 5,
    11: 5,
    12: 5,
    13: 5,
    14: 5,
    15: 5,
    16: 5,
    17: 4,
    18: 1,
    19: 0,
    20: 0,
    21: 0,
    22: 0,
    23: 0
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

    exit()



# ------------------------- END OF MAIN CODE ------------------------


if __name__ == '__main__':  main()

