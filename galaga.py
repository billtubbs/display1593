#!/usr/bin/python
# Python script to show galaga sprites on
# irregular LED display

import os
import time
import logging
from datetime import datetime
import numpy as np
import display1593 as display
import argparse
from collections import deque

parser = argparse.ArgumentParser(description='Display sprite images from'
                                             'Galaga arcade game.')
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
    6: 0,
    7: 1,
    8: 2,
    9: 4,
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

# [print(f'{np.exp(i):.1f}: {i},') for i in range(1, 12)]
b_levels = {
    2.7: 1,
    7.4: 2,
    20.1: 3,
    54.6: 4,
    148.4: 5,
    403: 6,
    1097: 7,
    2981: 8,
    8103: 9,
    22027: 10,
    59874: 11
}

fnames = [
    "galaga1.png",
    "galaga2.png",
    "galaga3.png",
    "galaga4.png",
    "galaga5.png",
    "galaga6.png",
    "galaga7.png",
    "galaga8.png",
    "galaga9.png",
    "galaga10.png",
    "galaga11.png",
    "galaga12.png",
    "galaga13.png",
    "galaga14.png",
    "galaga15.png",
    "galaga16.png",
    "galaga17.png",
    "galaga18.png",
    "galaga19.png",
    "galaga20.png",
    "galaga21.png",
    "galaga22.png",
    "galaga23.png",
    "galaga24.png",
    "galaga25.png",
    "galaga26.png"
]

image_dir = 'galaga/images'

def main():

    logging.info("\n\n----------------- galaga/main.py -----------------\n")

    dis = display.Display1593()
    dis.connect()
    dis.clear()

    i = 0
    status = ""

    # Get current time
    start_time = datetime.now()
    hr, mn, sc = (start_time.hour, start_time.minute, start_time.second)

    # Use an exponentially-weighted moving average
    ewma = dis.getBrightness()
    alpha = 0.1
    history = deque([ewma])
    history_length = 32

    brightness = bcycle[hr % 24]
    logging.info("Brightness: %d", brightness)
    logging.info("Delay time (mins): %d", args.period)

    wait_time = 60*args.period

    while True:
        for fname in fnames:

            logging.info("Showing image %s", fname.__repr__())

            # Set brightness level
            if brightness != bcycle[hr % 24]:
                brightness = bcycle[hr % 24]
                logging.info("Brightness: %d", brightness)

            filepath = os.path.join(image_dir, fname)
            dis.show_image_calibrated(filepath, brightness=brightness)

            while ((datetime.now() - start_time).total_seconds() < wait_time):
                time.sleep(1)

            # Get current time
            start_time = datetime.now()
            hr, mn, sc = (start_time.hour, start_time.minute, start_time.second)

    exit()



# ------------------------- END OF MAIN CODE ------------------------


if __name__ == '__main__':
    main()
