#!/usr/bin/python
"""------------------- shapes.py -------------------
Python module to display random shapes (squares,
circles, rectangles) that move slowly.

Author: Bill Tubbs
This was verion last updated: December 2017

Execute the module directly to display the current
time as follows:

$ python shapes.py

Note, shapes.py requires the following modules:
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
	filename='logfile.txt',
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


class Shape(object):
    """Class to simulate a shape"""

    next_id = 0

    @classmethod
    def update_next_id(cls):
        cls.next_id += 1

    def __init__(self, stype, (x, y), sizes, colour, speed,
                 direction, rotation=0):

        self.stype = stype
        self.x = x
        self.y = y
        self.sizes = sizes
        self.colour = colour
        self.speed = speed
        self.direction = direction
        self.rotation = rotation
        self.id = self.next_id
        self.update_next_id()

        now = datetime.now()
        logging.info("%s %d (%.0f, %.0f) created.", self.stype, self.id,
                     self.x, self.y)

    def display(self):

        if self.stype == "circle":
            # gfxdraw.aacircle(surface, x, y, r, color) -> None
            r = self.sizes[0]/2
            pygame.gfxdraw.aacircle(screen,
                int(self.x - r), int(self.y - r), int(r), self.colour)
            pygame.gfxdraw.filled_circle(screen,
                int(self.x - r), int(self.y - r), int(r), self.colour)

        elif self.stype == "square":
            # draw.rect(screen, color, (x,y,width,height), thickness)
            r = self.sizes[0]/2
            pygame.draw.rect(screen,
                self.colour, (int(self.x - r), int(self.y - r),
                self.sizes[0], self.sizes[0]), 0)

        elif self.stype == "rectangle":
            # draw.rect(screen, color, (x,y,width,height), thickness)
            w, h = self.sizes[0]/2, self.sizes[1]/2
            pygame.draw.rect(screen,
                self.colour, (int(self.x - w), int(self.y - h),
                self.sizes[0], self.sizes[1]), 0)

        elif self.stype == "strip":
            # draw.rect(screen, color, (x,y,width,height), thickness)
            if self.sizes[0] == 0:
                pygame.draw.rect(screen, self.colour, (-width, int(self.y),
                             width*3, self.sizes[1]), 0)
            elif self.sizes[1] == 0:
                pygame.draw.rect(screen, self.colour, (int(self.x), -height,
                             self.sizes[0], height*3), 0)

        elif self.stype == "triangle":
            # gfxdraw.filled_trigon(surface, x1, y1, x2, y2, x3, y3,
            #                       color) -> None

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
        if self.stype == 'strip':
            if self.sizes[0] == 0:
                self.y -= self.speed
            elif self.sizes[1] == 0:
                self.x += self.speed
        else:
            self.x += math.sin(self.direction) * self.speed
            self.y -= math.cos(self.direction) * self.speed
        if self.stype == "triangle":
            self.direction += self.rotation

    def check_in_bounds(self):
        """Return True if shape is within the bounds:
        (-width <= self.x < 2*width) and (-height <= self.y
        < 2*height).
        """
        return ((-width <= self.x < 2*width) and
                (-height <= self.y < 2*height))

    def __repr__(self):
        return "Shape({}, ({}, {}), ({}, {}), {}, {}, {})".format(
                    self.stype, self.x, self.y, self.sizes[0],
                    self.sizes[1], self.colour, self.speed,
                    self.direction)


class RandomShape(Shape):
    """Class to simulate a random shape"""

    def __init__(self):

        params = self.random_init_params()
        stype, (x, y), sizes, colour, speed, direction, rotation = params

        # Python 2 version:
        super(self.__class__, self).__init__(stype, (x, y), sizes, colour,
                                             speed, direction, rotation)

        # Python 3 version:
        #super().__init__(stype, (x, y), size, colour, speed, direction,
        #                 rotation)

    def random_init_params(self):

        #probs = [0.2, 0.2, 0.2, 0.3, 0.1]
        stype = random.choice(shape_types + shape_types[:-1]) # less triangles

        a, b = (random.randint(2, 5), random.randint(2, 5))
        c = random.randint(abs(b - a) + 1, a + b - 1)
        size = random.uniform(0.2, 1.2)*math.sqrt(width*height)
        sizes = [s*size/5 for s in (a, b, c)]

        # Position object somewhere on boundary:
        #r = [random.randint(0, 1), random.random()]
        #random.shuffle(r)
        #x = int(3*width*r[0]) - width
        #y = int(3*height*r[1]) - height

        # Position objects randomly across available space
        x = width*random.uniform(-1.0, 2.0)
        y = height*random.uniform(-1.0, 2.0)

        if stype == "strip":
            if x in (-width, width*2):
                x = width*0.5
                sizes[0] = 0
                sizes[1] = random.randint(7, 16)
                while 0 < y < height:
                    y = int((random.random()*3 - 1)*height)
            if y in (-height, height*2):
                y = height*0.5
                sizes[1] = 0
                sizes[0] = random.randint(7, 16)
                while 0 < x < width:
                    x = int((random.random()*3 - 1)*width)
        colour = random.choice(colours)
        speed = random.uniform(0.05, 0.25)
        direction = random.uniform(0, math.pi*2)

        if stype == "triangle":
            rotation = random.uniform(-math.pi/720, math.pi/720)
        else:
            rotation = 0

        return (stype, (x, y), sizes, colour, speed, direction, rotation)


logging.info("\n\n-------------- shapes.py --------------\n")
logging.info("Display slowly moving random shapes.")
logging.info("Screen brightness: %f", brightness)

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

number_of_shapes = 10
my_shapes = []
logging.info("Initializing %d shapes...", number_of_shapes)
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
        if shape.check_in_bounds() is not True:
            logging.info("%s %d (%.0f, %.0f) out of bounds.",
                         shape.stype, shape.id, shape.x, shape.y)
            # Create new shape
            my_shapes[i] = RandomShape()
            shape = my_shapes[i]
        shape.display()

    # Update display screen
    #pygame.display.flip()

    # Save image to disk
    pygame.image.save(screen, "shapes.png")

    # Tell display to show image
    dis.show_image("shapes.png")

    clock.tick(0.5)