#!/usr/bin/python
"""------------------- shapes.py -------------------
Python module to display brightness level
according to photo resistor.

Author: Bill Tubbs
This was verion last updated: December 2017

Pygame code is based on a tutorial from
http://www.petercollingridge.co.uk/pygame-physics-simulation
"""

import logging
import display1593 as display
import pygame
import pygame.gfxdraw
from datetime import datetime
from collections import deque


logging.basicConfig(
	filename='logfile.txt',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def colour(value, brightness=1.0):
    return (
        int(float(value >> 16 & 0xff)*brightness),
        int(float(value >> 8 & 0xff)*brightness),
        int(float(value & 0xff)*brightness)
    )

background_colour = (0, 0, 0)
line_colour = colour(0x377eb8, 0.5)

# Colours from colorbrewer2.org:
# http://colorbrewer2.org/#type=qualitative&scheme=Dark2&n=8
colour_values = [
    0x1b9e77,
	0xd95f02,
	0x7570b3,
	0xe7298a,
	0x66a61e,
	0xe6ab02,
	0xa6761d,
	0x666666
]

colour_values = [
    0xe41a1c,
	0x377eb8,
	0x4daf4a,
	0x984ea3,
	0xff7f00,
	0xffff33,
	0xa65628,
	0xf781bf
]

width, height = (256, 256)

logging.info("\n\n-------------- lightlevels.py --------------\n")
logging.info("Display brightness reading from photo resistor.")

dis = display.Display1593()
dis.connect()
dis.clear()

# Use this to initialize a pygame window
#screen = pygame.display.set_mode((width, height))
#pygame.display.set_caption('Shapes')

# Use this if you do not want to have a visible window
logging.info("Starting pygame...")
screen = pygame.Surface((width, height))
clock = pygame.time.Clock()

# Initialise variables
running = True

# Output filename
filename = "lightlevels.csv"

# Use an exponentially-weighted moving average
ewma = dis.getBrightness()
alpha = 0.1
history = deque([ewma])
history_length = 32

# Main operations loop
while running:

    now = datetime.now()

    # Check for window events
    brightness = dis.getBrightness()
    logging.info("Brightness: %d", brightness)
    print("Brightness: %d" % brightness)
    ewma = alpha*brightness + (1 - alpha)*ewma

    # Clear screen
    screen.fill(background_colour)

    # Process and display shapes
    for i, value in enumerate(history):
        pygame.draw.rect(
            screen,
            line_colour,
            (i*width/history_length, height - value - 6,
             width/history_length, 6),
            0
        )

    # Update display screen
    #pygame.display.flip()

    # Convert pygame screen to numpy array
    s = screen.get_buffer()
    x = pygame.surfarray.pixels3d(screen).swapaxes(0, 1)

    # Tell display to show image
    z = dis.convert_image(x)
    dis.setAllLeds(z)

    history.append(ewma)
    if len(history) > history_length:
        history.popleft()

    sec = now.second
    if (sec % 5) == 0:
        time_string = now.strftime("%Y-%m-%d %H:%M:%S")
        output = "{0:s}, {1:.1f}".format(time_string, ewma)
        with open(filename, "a") as f:
            f.write(output + '\n')

    clock.tick(0.5)