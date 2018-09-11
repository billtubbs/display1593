#!/usr/bin/python
"""------------------- shapes.py -------------------
Python module to display random shapes (squares,
circles, rectangles) that move slowly.

Author: Bill Tubbs
This was verion last updated: December 2017

Execute the module directly to display the current
time as follows:

$ python shapes.py

Note, shapes.py requires the following components:
- display1593

Pygame code is based on a tutorial from
http://www.petercollingridge.co.uk/pygame-physics-simulation
"""

import logging
import display1593 as display
import pygame
import pygame.gfxdraw
import random
import math
from datetime import datetime

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

background_colour = (0, 0, 0)

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

def colour(value, brightness=1.0):
    return (
        int(float(value >> 16 & 0xff)*brightness),
        int(float(value >> 8 & 0xff)*brightness),
        int(float(value & 0xff)*brightness)
    )

brightness = 0.5
colours = [colour(v, brightness) for v in colour_values]

width, height = (256, 256)
shape_types = ["circle", "square", "rectangle", "strip", "triangle"]


class Shape():
    """Class to simulate a shape"""

    def __init__(self, type, (x, y), size, colour, speed,
                 direction, rotation=0):
        self.type = type
        self.x = x
        self.y = y
        self.sizes = sizes
        self.colour = colour
        self.speed = speed
        self.direction = direction
        self.rotation = rotation

    def display(self):

        if self.type == "circle":
            # gfxdraw.aacircle(surface, x, y, r, color) -> None
            r = self.sizes[0]/2
            pygame.gfxdraw.aacircle(screen,
                int(self.x - r), int(self.y - r), int(r), self.colour)
            pygame.gfxdraw.filled_circle(screen,
                int(self.x - r), int(self.y - r), int(r), self.colour)

        elif self.type == "square":
            # draw.rect(screen, color, (x,y,width,height), thickness)
            r = self.sizes[0]/2
            pygame.draw.rect(screen,
                self.colour, (int(self.x - r), int(self.y - r),
                self.sizes[0], self.sizes[0]), 0)

        elif self.type == "rectangle":
            # draw.rect(screen, color, (x,y,width,height), thickness)
            w, h = self.sizes[0]/2, self.sizes[1]/2
            pygame.draw.rect(screen,
                self.colour, (int(self.x - w), int(self.y - h),
                self.sizes[0], self.sizes[1]), 0)

        elif self.type == "strip":
            # draw.rect(screen, color, (x,y,width,height), thickness)
            if self.sizes[0] == 0:
                pygame.draw.rect(screen, self.colour, (-width, int(self.y),
                             width*3, self.sizes[1]), 0)
            elif self.sizes[1] == 0:
                pygame.draw.rect(screen, self.colour, (int(self.x), -height,
                             self.sizes[0], height*3), 0)

        elif self.type == "triangle":
            # gfxdraw.filled_trigon(surface, x1, y1, x2, y2, x3, y3, color) -> None
            a, b, c = self.sizes[0:3]

            tb = math.acos((a*a + c*c - b*b)/(2*a*c))

            rel_pts = (
                (-a*0.5, 0),
                (a*0.5, 0),
                (c*math.cos(tb) - a*0.5, c*math.sin(tb))
            )

            points = []
            for p in rel_pts:
                if p[0] == 0:
                    r = p[1]
                    theta = 0.5*math.pi
                else:
                    theta = math.atan(p[1]/p[0])
                r = p[0]/math.cos(theta)
                theta += self.direction
                points.append((self.x + r*math.sin(theta),
                               self.y + r*math.cos(theta)))

            p1, p2, p3 = points
            pygame.gfxdraw.filled_trigon(screen, int(p1[0]), int(p1[1]),
                                         int(p2[0]), int(p2[1]),
                                         int(p3[0]), int(p3[1]),
                                         self.colour)

    def move(self):
        if self.type == 'strip':
            if self.sizes[0] == 0:
                self.y -= self.speed
            elif self.sizes[1] == 0:
                self.x += self.speed
        else:
            self.x += math.sin(self.direction) * self.speed
            self.y -= math.cos(self.direction) * self.speed
        if self.type == "triangle":
            self.direction += self.rotation

    def __repr__(self):
        return "Shape({}, ({}, {}), ({}, {}), {}, {}, {})".format(self.type, self.x, self.y, self.sizes[0], self.sizes[1], self.colour, self.speed, self.direction)

class RandomShape(Shape):
    """Class to simulate a random shape"""

    def __init__(self):
        self.random_init()

    def random_init(self):

        self.type = random.choice(shape_types)

        a, b = (random.randint(2, 5), random.randint(2, 5))
        c = random.randint(abs(b - a) + 1, a + b - 1)
        size = random.uniform(0.2, 1.2)*math.sqrt(width*height)
        self.sizes = [s*size/5 for s in (a, b, c)]

        # Position object somewhere on boundary:
        r = [random.randint(0, 1), random.random()]
        random.shuffle(r)
        self.x = int(3*width*r[0]) - width
        self.y = int(3*height*r[1]) - height

        # Alternatively, position object near centre of screen
        #self.x = width*random.uniform(0.2, 0.8)
        #self.y = height*random.uniform(0.2, 0.8)

        if self.type == "strip":
            if self.x in (-width, width*2):
                self.x = width*0.5
                self.sizes[0] = 0
                self.sizes[1] = random.randint(7, 16)
                while 0 < self.y < height:
                    self.y = int((random.random()*3 - 1)*height)
            if self.y in (-height, height*2):
                self.y = height*0.5
                self.sizes[1] = 0
                self.sizes[0] = random.randint(7, 16)
                while 0 < self.x < width:
                    self.x = int((random.random()*3 - 1)*width)
        self.colour = random.choice(colours)
        self.speed = random.uniform(0.1, 0.3333)
        self.direction = random.uniform(0, math.pi*2)

        if self.type == "triangle":
            self.rotation = random.uniform(-math.pi/720, math.pi/720)
        else:
            self.rotation = 0

        now = datetime.now()
        logging.info("%s (%.0f, %.0f) created.", self.type, self.x, self.y)

    def check_bounds(self):
        if (self.x < -width or self.x > 2*width or
            self.y < -height or self.y > 2*height):
             logging.info("%s (%.0f, %.0f) removed.", self.type, self.x,
                          self.y)
             self.random_init()


logging.info("\n\n-------------- shapes.py --------------\n")
logging.info("Display slowly moving random shapes.")

dis = display.Display1593()
dis.connect()
dis.clear()


# Use this to initialize a pygame window
#screen = pygame.display.set_mode((width, height))
#pygame.display.set_caption('Shapes')

# Use this if you do not want to have a visible window
screen = pygame.Surface((width, height))
clock = pygame.time.Clock()

number_of_shapes = 10
my_shapes = []

for n in range(number_of_shapes):
    my_shapes.append(RandomShape())

# Initialise variables
running = True
selected_particle = None

# Main operations loop
while running:

    # Check for window events
    #for event in pygame.event.get():
    #    if event.type == pygame.QUIT:
    #        running = False

    # Clear screen
    screen.fill(background_colour)

    # Process and display particles
    for i, shape in enumerate(my_shapes):
        shape.move()
        shape.check_bounds()
        shape.display()

    # Update display screen
    #pygame.display.flip()

    # Save image to disk
    pygame.image.save(screen, "shapes.png")

    # Tell display to show image
    dis.show_image("shapes.png")

    clock.tick(0.5)