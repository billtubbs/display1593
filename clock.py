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


from PIL import Image, ImageDraw
import numpy as np
import sys
from datetime import datetime
import display1593 as display

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

dark_red = red*2/3
dark_grn = grn*2/3
dark_blu = blu*2/3
grey = white*2/3

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


def main(argv):
    """Demonstration of how to use this module."""

    print "\n------------- Analog Clock Display -------------\n"
    print "Displays current time as an analog clock face."

    dis = display.Display1593()
    dis.connect()
    dis.clear()

    # Get current time
    t = datetime.now().time()
    hr, min = (t.hour, t.minute)

    print "Displaying clock..."

    try:
        color = argv[0]
    except IndexError:
        color = 'r'

    try:
        imax = int(argv[1])
    except IndexError:
        imax = 200

    img = clock_image(t)
    img.save("clock.png")
    dis.show_image("clock.png")

    hr, min, s = t.hour, t.minute, t.second

    while True:
        print "%02d:%02d" % (hr, min)

        while t.minute == min:
            while datetime.now().time().second == s:
                pass

            t = datetime.now().time()

        img = clock_image(t)
        img.save("clock.png")
        dis.show_image("clock.png")

        min = t.minute
        if min == 0:
            hr = t.hour


if __name__ == "__main__":
    main(sys.argv[1:])
