#!/usr/bin/python
# Python script for irregular 1593 LED array
# Randomly displays stars and then they slowly
# fade away...

import logging
from datetime import datetime
import numpy as np

import display1593 as display

logging.basicConfig(
	filename='logfile.txt',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logging.info("\n\n------------ Fading Stars Algorithm ------------\n")

dis = display.Display1593()
dis.connect()
dis.clear()

stars = []
non_stars = list(range(display.leds.numCells))

# Must be a float greater than 1.0
fade_rate = 1.018

# Best in range to 256 to 1024
initial_brightness = 750

# Max number of stars
n_max = 64

# Time increment (seconds) determines speed
# of updating (suggest 0.25 to 10 seconds)
time_step = 2.0

t = datetime.now()
minute = t.minute

logging.info("Beginning to show stars...")

while True:

    if len(stars) < n_max:
        if np.random.randint(8) == 0:
            new_star = np.random.choice(non_stars)
            stars.append([
                    new_star,
                    float(initial_brightness*(0.5 + 0.5*np.random.random()))
                ]
            )
            non_stars.remove(new_star)

    for p in stars:
        p[1] = p[1]/fade_rate
        if p[1] < 3.5:
            stars.remove(p)
            non_stars.append(p[0])
            dis.setLed(p[0], [0]*3)
        else:
            dis.setLed(p[0], [int(p[1]/8 + 0.5)]*3)


    while (datetime.now() - t).total_seconds() < time_step:
        pass
    t = datetime.now()

    if minute != t.minute:
        logging.info(len(stars))
        minute = t.minute