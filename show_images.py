#!/usr/bin/python
# Python code for RaspberryPi to communicate with
# Teensy driving the LED display
#
# See here:
# http://blog.oscarliang.net/connect-raspberry-pi-and-arduino-usb-cable/

import logging
from datetime import datetime
import numpy as np
import display1593 as display
import argparse

parser = argparse.ArgumentParser(description='Display sequence of images.')
parser.add_argument('period', type=int, default=20,
                    help='Time in minutes that each image is displayed.')
args = parser.parse_args()

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
    6: 1,
    7: 2,
    8: 4,
    9: 5,
    10: 5,
    11: 5,
    12: 5,
    13: 5,
    14: 5,
    15: 5,
    16: 5,
    17: 4,
    18: 2,
    19: 1,
    20: 1,
    21: 1,
    22: 1,
    23: 0
}


def main():

    logging.info("\n\n------------------- show_images.py -------------------\n")

    dis = display.Display1593()
    dis.connect()

    dis.clear()

    fnames = [
        "images/monalisa.png",
        "images/einstein.jpeg",
        "images/audrey.jpg",
        "images/muhammedali.jpg",
        "images/bowie.jpg",
        "images/spock.jpg",
        "images/hendrix.jpg",
        "images/amy.png",
        "images/cohen.png"
    ]

    i = 0
    status = ""

    # Get current time
    start_time = datetime.now()
    hr, mn, sc = (start_time.hour, start_time.minute, start_time.second)

    brightness = bcycle[hr % 24]
    logging.info("Brightness: %d", brightness)
    logging.info("Delay time (mins): %d", args.period)

    while True:

        for f in fnames:

            logging.info("Showing image %s", f.__repr__())

            # Set brightness level
            if brightness != bcycle[hr % 24]:
                brightness = bcycle[hr % 24]
                logging.info("Brightness adjusted: %d", brightness)

            dis.show_image_calibrated(f, brightness=brightness)

            while ((datetime.now() - start_time).total_seconds() <
                   60*args.period):
                pass

            # Get current time
            start_time = datetime.now()
            hr, mn, sc = (start_time.hour, start_time.minute, start_time.second)

    exit()



# ------------------------- END OF MAIN CODE ------------------------


if __name__ == '__main__':  main()

