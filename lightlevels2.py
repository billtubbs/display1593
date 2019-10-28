#!/usr/bin/python
"""------------------- lightlevels.py -------------------
Python module to display brightness level according to
photo resistor.
"""

import logging
import display1593 as display
import pygame
import pygame.gfxdraw
from datetime import datetime
from collections import deque
import time

logging.basicConfig(
	filename='logfile.txt',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def display_digits(n):
    if n >= 0:
        digit_str = "{:4d}".format(int(n))
    else:
        digit_str = "  :  "
    digit_str = "{:2s} {:2s}".format(digit_str[-4:-2], digit_str[-2:])
    dclock.display_time(digit_str)


from digclock import DigitalClock

dclock = DigitalClock(imax=32)

# Get current time
t = datetime.now()
date_string = datetime.strftime(t, "%Y-%m-%d")

# Output filename
filename = "lightlevels-{}.csv".format(date_string)
logging.info("Readings will be appended to '%s'.", filename)

# EWMA parameter
alpha = 0.1
ewma = dclock.dis.getBrightness()

running = True
while running:
    now = datetime.now()
    # Read light level from photo resistor
    brightness = dclock.dis.getBrightness()
    ewma = alpha * brightness + (1 - alpha) * ewma
    logging.info("Brightness: %d", brightness)
    display_digits(int(ewma))
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    output = "{0:s},{1:d},{2:.2f}".format(timestamp, brightness, ewma)
    with open(filename, "a") as f:
            f.write(output + '\n')
    sec = now.second
    while (datetime.now().second == sec):
        time.sleep(0.01)
