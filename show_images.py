#!/usr/bin/python
# Python code for RaspberryPi to communicate with
# Teensy driving the LED display
#
# See here:
# http://blog.oscarliang.net/connect-raspberry-pi-and-arduino-usb-cable/

import logging
from datetime import datetime
import numpy as np
from scipy import ndimage
import display1593 as display

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# To identify new devices connected to Raspberry Pi USB ports
# Use type lsusb into the shell
# Example:
# Bus 001 Device 006: ID 16c0:0483 VOTI Teensyduino Serial
# Bus 001 Device 005: ID 16c0:0483 VOTI Teensyduino Serial

# To find the port name of each Teensy use the following with
# and without the Teensies plugged in:
# ls /dev/tty*


bcycle = {
    0: 12,
    1: 12,
    2: 12,
    3: 12,
    4: 12,
    5: 12,
    6: 12,
    7: 10,
    8: 5,
    9: 3,
    10: 2,
    11: 2,
    12: 2,
    13: 2,
    14: 2,
    15: 2,
    16: 2,
    17: 3,
    18: 5,
    19: 10,
    20: 12,
    21: 12,
    22: 12,
    23: 12
}


def main():

    logging.info("\n\n------------------- show_images.py -------------------\n")

    dis = display.Display1593()
    dis.connect()

    dis.clear()

    fnames = [
        "images/monalisa.png",
        "images/einstein.jpeg",
        "images/muhammedali.jpg",
        "images/bowie.jpg",
        "images/spock.jpg",
        "images/hendrix.jpg"
    ]

    i = 0
    status = ""
    col = 0
    prev_dimness = None

    while True:

        for f in fnames:

            logging.info("Showing image %s", f.__repr__())
            # Get current time
            start_time = datetime.now()
            hr, mn, sc = (start_time.hour, start_time.minute, start_time.second)

            # Set display dimmer level
            dimness = bcycle[hr % 24]
            if dimness != prev_dimness:
                logging.info("Dimness adjusted: %d", dimness)

            dis.show_image(f, dimness=dimness)

            while (datetime.now() - start_time).total_seconds() < 60*10:
                pass

    exit()



# ------------------------- END OF MAIN CODE ------------------------


if __name__ == '__main__':  main()

