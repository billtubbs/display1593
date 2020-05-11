#!/usr/bin/python
"""------------- Analog Clock Display -------------
Python module to display the time as an analog clock
face similar to a mechanical clock used at railway
stations. Uses the Irregular 1593-LED display built
by me.

Author: Bill Tubbs
This was verion last updated: November 2017

Execute the module directly to display the current
time as follows:

$ python clock.py

Note, clock.py requires the following components:
- PIL
- numpy
- display1593
"""


import logging
import sys
from datetime import datetime
from PIL import Image, ImageDraw
import numpy as np
import display1593 as display

logging.basicConfig(
	filename='logfile.txt',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


# For python 2 compatibility:
try:
    FileNotFoundError = IOError
except:
    pass

display_size = (256, 256)

r, g, b = 200, 200, 160

blk = np.array((0, 0, 0))
red = np.array((r, 0, 0))
grn = np.array((0, g, 0))
blu = np.array((0, 0, b))
white = np.array((r, g, b))

dark_red = red*2//3
dark_grn = grn*2//3
dark_blu = blu*2//3
grey = white*2//3

size = np.array(display_size)
centre = size/2

colors = (tuple(grey), tuple(white))

# Dimensions of hour markers and hands
r1, r2, r3, r4 = 114, 128, 123, 80
w = 6

def radial_line(centre, theta, r1, r2, w):

    e = np.array((-w*np.cos(theta), -w*np.sin(theta)))
    p1 = np.array((centre[0] + r1*np.sin(theta), centre[1] - r1*np.cos(theta)))
    p2 = np.array((centre[0] + r2*np.sin(theta), centre[1] - r2*np.cos(theta)))

    return (
        tuple(p1 + e),
        tuple(p2 + e),
        tuple(p2 - e),
        tuple(p1 - e)
    )

def clock_image(time):

    # Create new image
    display = Image.new('RGBA', display_size, color=0)

    # Draw graphics
    draw = ImageDraw.Draw(display)

    # Add hour markers
    for angle in range(0, 360, 30):
        theta = np.radians(angle)
        points = radial_line(centre, theta, r1, r2, w)
        draw.polygon(points, fill=colors[0])

    # Add minute and hour hands
    h1_angle = np.radians(time.minute*360/60)
    h2_angle = np.radians((time.hour*360 + time.minute*6)/12)

    points = radial_line(centre, h1_angle, -w, r3, w)
    draw.polygon(points, fill=colors[1])

    points = radial_line(centre, h2_angle, -w, r4, w+2)
    draw.polygon(points, fill=colors[1])

    return display

def show_image_inc(dis, img_old, img_new):

    img_array1 = np.array(img_old)
    img_array2 = np.array(img_new)
    assert img_array1.shape == (256, 256, 4)
    assert img_array2.shape == (256, 256, 4)
    z1 = dis.convert_image(img_array1)[:,:3]
    z2 = dis.convert_image(img_array2)[:,:3]
    difference = z2 - z1
    diff_idx = difference.sum(axis=1).nonzero()

    leds = diff_idx[0]
    cols = z2[diff_idx[0]]

    # Use rgb_scales array to calibrate intensities
#     rgb = np.array([0, 1, 2]*leds.numCells)
#     z = rgb_scales[brightness - 1][
#         (z.ravel()//8, rgb)
#     ].reshape(leds.numCells, 3)

    for i, led in enumerate(leds):
        dis.setLed(led, (cols[i,:] // 10).tolist())

    #TODO: This doesn't work - comm errors
    #dis.setLeds(diff_idx[0], z2[diff_idx])


def main(argv):
    """Demonstration of how to use this module."""

    logging.info("\n\n------------- Analog Clock Display -------------\n")
    logging.info("Displays current time as an analog clock face.")

    dis = display.Display1593()
    dis.connect()
    dis.clear()

    # Get current time
    t = datetime.now().time()
    hr, min = (t.hour, t.minute)

    logging.info("Displaying clock...")

    try:
        color = argv[0]
    except IndexError:
        color = 'r'

    try:
        imax = int(argv[1])
    except IndexError:
        imax = 200

    img_old = Image.fromarray(np.zeros((256, 256, 4), dtype='uint8'), 'RGBA')
    img_new = clock_image(t)
    show_image_inc(dis, img_old, img_new)
    img_old = img_new

    hr, min, s = t.hour, t.minute, t.second

    while True:
        logging.info("%02d:%02d" % (hr, min))

        while t.minute == min:
            while datetime.now().time().second == s:
                pass
            t = datetime.now().time()

        img_new = clock_image(t)
        show_image_inc(dis, img_old, img_new)
        img_old = img_new

        min = t.minute
        if min == 0:
            hr = t.hour


if __name__ == "__main__":
    main(sys.argv[1:])
