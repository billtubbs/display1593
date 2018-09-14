#!/usr/bin/python
# ---------- Irregular LED Display Emulator module ----------
# This module written in Python contains class definitions
# and methods to emulate an irregular array of 1593 LEDs
# designed and built by Bill Tubbs in 2014-2015.

# The purpose of this code is to develop and test display
# projects before transporting them to the Raspberry Pi
# connected to the display.
#
# - display1593.py is the real display driver module.


import os
import pickle
import logging
import numpy as np
from scipy import ndimage
from scipy import misc
from itertools import izip
import pygame
import pygame.gfxdraw
from pygame import KEYDOWN, K_ESCAPE, QUIT


logging.basicConfig(
    filename='logfile.txt',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Load irregular LED array data
import ledArray_data_1593 as leds

f = open('data/mask1593.pickle', 'r')
mask1593 = pickle.load(f)   # load object from file
f.close()

addresses = ('/dev/ttyACM0', '/dev/ttyACM1')


class Display1593Error(Exception):
    """Exception class for reporting display1593 errors."""

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class Display1593EmulatorWindow(object):

    def __init__(self, led_state, size=512):

        # Pygame parameters
        self.led_state = led_state
        self.hscale = float(size)/leds.width
        self.vscale = float(size)/leds.height
        self.led_positions = np.vstack((leds.centres_x*self.hscale,
                                        leds.centres_y*self.vscale)).transpose().astype(int)
        self.led_size = int(24*self.hscale)
        self.background_colour = (0, 0, 0)

        # Set up pygame window
        pygame.init()
        self.screen = pygame.display.set_mode((size, size))
        pygame.display.set_caption('Irregular Led Display')

        self.background = pygame.Surface(self.screen.get_size())
        self.background = self.background.convert()
        self.background.fill(self.background_colour)

        # Blit everything to the screen
        self.screen.blit(self.background, (0, 0))
        pygame.display.flip()

        self.update()
        self.clock = pygame.time.Clock()


    def update(self):

        self.screen.blit(self.background, (0, 0))
        pygame.event.get()
        high_intensity = ((16777216*self.led_state)**0.25).astype('B')

        for led_id in range(leds.numCells):
            col = high_intensity[led_id]
            #import pdb; pdb.set_trace()
            pygame.draw.circle(self.screen, col, self.led_positions[led_id], self.led_size, 0)
            #pygame.gfxdraw.filled_circle(self.screen, self.led_positions[led_id, 0],
            #                             self.led_positions[led_id, 1], self.led_size, col)

        # Display screen
        pygame.display.flip()


class Teensy(object):
    """Teensy micro-controller connection object class."""

    teensy_names = ["Teensy1", "Teensy2"]

    def __init__(self, address, baud=38400):

        self.address = address
        self.baud = baud
        self.serial = None
        self.name = None

    def connect(self):

        # Establish (pretend) connection
        self.serial = "serial.Serial(%s, %s)" % (self.address, self.baud)
        #self.serial.open()

        # Request identification
        #self.serial.write('ID\n')

        result = False
        # Fake a connection
        response = self.teensy_names.pop()

        if response.startswith("Teensy"):
            self.name = response
            logging.info("Connection to %s (%s) successful.",
                         self.address, response)
            result = True

        return result

    def __repr__(self):

        return "Teensy('%s', baud=%d)" % (self.address, self.baud)


class Display1593(object):
    """Irregular LED array driver class."""

    def __init__(self, size=512):

        self.tys = None
        self.size = size
        self.state_filename = "led_state.pickle"

        # Load LED current state from file
        if self.state_filename in os.listdir("."):
            with open(self.state_filename, 'rb') as f:
                self.led_state = pickle.load(f)   # load object from file
            assert self.led_state.dtype == 'B'
        else:
            self.led_state = np.zeros((1593, 3), dtype='B')

        self.window = Display1593EmulatorWindow(self.led_state, self.size)

    def connect(self, addresses=addresses):

        logging.info("Finding Teensies...")
        find_these = [address.split('/')[-1] for address in addresses]

        tty_files = []
        # Make it look like the right Teensies were found!
        for filename in find_these:
            if filename.startswith('ttyACM'):
                tty_files.append(filename)

        logging.info("Trying to connect to Teensies...")

        self.tys = []
        for a in tty_files[0:2]:
            self.tys.append(Teensy('/dev/' + a))

        if len(self.tys) != 2:
            raise Display1593Error("Could not find two Teensies to connect to.")

        for t in self.tys:
            t.connect()

        if self.tys[1].name == 'Teensy1':
            self.tys[0], self.tys[1] = (self.tys[1], self.tys[0])
            logging.info("(Teensies swapped)")

        if (self.tys[0].name, self.tys[1].name) != ('Teensy1', 'Teensy2'):
            raise Display1593Error(
                "Could not find the right pair of Teensy controllers. "
                "Found these instead: %s, %s" % (
                    self.tys[0].name,
                    self.tys[1].name
                )
            )

    def setLed(self, led, col):
        """Set colours of an individual LED."""

        self.led_state[led] = col
        self.window.update()

        #logging.info("Led %d set to 0x%2x%2x%2x", led, col[0], col[1], col[2])

    def setLeds(self, ledIDs, cols):
        """setLeds(self, ledIDs, cols)

        Set colours of a batch of LEDs.

        Arg:
            ledIDs: list or tuple containing the LED ID numbers
            cols: list or tuple containing the LED colour
                intensities (r, g, b).
        """

        if len(cols) != len(ledIDs):
            raise Display1593Error("ledIDs and cols must be sequences"
                                   " of equal length.")

        self.led_state[ledIDs] = cols
        self.window.update()

        #logging.info("%d leds set", len(ledIDs))

    def setAllLeds(self, cols):
        """setAllLeds(self, cols)
        Set all LEDs to the specified sequence of colours.
        Args:
            cols: sequence of 1593 (r, g, b) values or an
                array of values of shape (1593, 3).
        """

        if len(cols) != leds.numCells:
            raise Display1593Error("setAllLeds() requires a sequence of"
                                   " %d colours." % leds.numCells)

        self.led_state[:] = cols
        self.window.update()

        #logging.info("All leds set")

    @staticmethod
    def send_chars(controller, char_data):

        n = controller.serial.write(char_data)

        if n != len(char_data):
            raise Display1593Error(
                  "Error sending data to Teensy. "
                  "{} bytes sent out of {}.".format(n, len(char_data))
                  )

    def clear(self):
        """Set all LEDs to black (0, 0, 0)."""

        self.led_state[:] = 0
        self.window.update()
        #logging.info("Display cleared")

    def getBrightness(self):
        """getBrightness() -> int

        Gets and returns the analog reading from the
        photo-resistor connected to Teensy 1. This value
        should be an integer in the range 0 to 1023."""

        for controller in self.tys:
            if controller.name == 'Teensy1':
                controller.serial.write('B')
                data = controller.serial.read(size=2)
                return (ord(data[0]) << 8) + ord(data[1])

    def getColour(self, led):
        """getColour(led) -> (int, int, int)

        Get the colour of an individual LED."""

        led_ref = leds.ledIndex[led]
        controller = self.tys[led_ref[0]]
        i = led_ref[1]
        controller.serial.write('G' + chr(i >> 8) + chr(i % 256))
        col = controller.serial.read(size=3)
        return ord(col[0]), ord(col[1]), ord(col[2])

    def prepare_image(self, image):
        """Adjusts the size of the ndimage stored in image and
        returns a 256x256 pixel image suitable for use with
        the method convert_image()."""

        #logging.info("Preparing image %s", str(image.shape))

        # Find out how many layers the image has
        image_shape = image.shape[:2]
        if len(image.shape) > 2:
            image_cols = image.shape[2]
        else:
            image_cols = 1

        data_type = image.dtype

        # Reduce redundant layers - if imageCols > 1,
        # assume first three are red, green, blue and
        # ignore the rest

        if image_cols > 3:
            image = image[:, :, :3]
            image_cols = image.shape[2]
            #logging.info("Image reduced to %d layers (R, G, B)", imageCols)

        # Determine smallest dimension (x or y)
        image_size = min(image_shape)

        if not image_shape[0] == image_shape[1]:
            if image_shape[0] > image_size:
                #logging.info("Image is not square. Y-axis will be cropped.")
                y = (image_shape[0] - image_size)/2
                image = image[:][y:y+image_size]
            elif image_shape[1] > image_size:
                #logging.info("Image is not square. X-axis will be cropped.")
                x = (image_shape[1] - image_size)/2
                image = image[:, x:x+image_size, :]

            #logging.info("Cropped image size: %dx%d", image_shape[0],
            #             image_shape[1])

        # Define arrays for maximum colour intensities
        # of final image

        cols = ['red', 'green', 'blue']

        if image_cols > 1:
            max_int = np.zeros((image_cols), dtype ='uint8')
            for i in range(image_cols):
                selCol = np.array([i], dtype=np.intp)
                colMin = image[:, :, selCol].min()
                colMax = image[:, :, selCol].max()
                #logging.info("Colour %d range: %d to %d", i, colMin, colMax)
                max_int[i] = max(255, colMax)
        else:
            logging.info("This is a black and white image.")
            logging.info("Intensity range: %d to %d", image.min(), image.max())
            logging.info("Specify RGB intensities for final image (0-31):")

            max_int = np.zeros((3), dtype ='uint8')
            for i, c in enumerate(cols):
                while True:
                    s = raw_input(c + ": ")
                    try:
                        f = float(s)
                    except:
                        print("Enter a number between 0 and 255")
                        continue
                    max_int[i] = f
                    break
        # Re-size image to required pixel size for super-sampling
        # algorithm

        pixels = 256

        # Re-size image for super-sampling
        s_image = misc.imresize(
            image,
            (pixels, pixels),
            interp='bilinear',
            mode=None
        )
        #logging.info("Image re-sized: %s", str(s_image.shape[:2]))

        return s_image

    def convert_image(self, image_array):
        """Convert 256 x 256 RGB image array to 1593 RGB led intensities."""

        global mask1593
        shape = image_array.shape
        img_data = image_array.reshape(shape[0]*shape[1], shape[2])
        return np.mean(img_data[mask1593], axis=1).astype(int)

    def show_image(self, filename, dimness=8):

        img = ndimage.imread(filename)
        z = self.convert_image(self.prepare_image(img))
        self.setAllLeds(z**2/(256*dimness))

    def camera_grab(self, filename):
        """Use the Picamera to take and image and save it
        to a file in the current working directory."""

        logging.info("Taking image with Picamera...")

        raise NotImplementedError()
        #with picamera.PiCamera() as camera:
        #    camera.resolution = (1024, 768)
        #    camera.start_preview()
        #    # Camera warm-up time
        #    time.sleep(2)
        #    camera.capture(filename)

        #logging.info("Image saved to file", filename)

    @staticmethod
    def image_snapshot(image, point, size):
        x = point[0]
        y = point[1]
        width = size[0]
        height = size[1]
        return image[y-height/2:y+height/2, x-width/2:x+width/2, :]

    def save_state(self):
        """Save led state"""

        with open(self.state_filename, 'wb') as f:
            pickle.dump(self.led_state, f)
        logging.info("Led state saved.")

    def __repr__(self):

        return "Display1593()"


class TestResults(object):

    def __init__(self):

        self.test_id = 0
        self.test_results = {}
        self.test_names = {}

    def record(self, name, result, announce=True):
        self.test_names[self.test_id] = name
        self.test_results[self.test_id] = result
        if announce:
            logging.info("Test %d %s: %s", self.test_id, name, str(result))
        self.test_id += 1


def run_tests():

    logging.info("\n\n ------------- display1593.py -------------")
    logging.info("Running tests of functions...")

    test_results = TestResults()

    dis = Display1593()
    test_results.record("Display1593 initialized", True)

    dis.connect()
    test_results.record("Connection to display", True)

    dis.clear()
    test_results.record("Clear display", True)

    for i in range(25):
        #pygame.event.get()
        dis.setLed(i, tuple(np.random.randint(0, 128, size=3)))
        dis.window.clock.tick(30)

    test_results.record("setLed()", True)
    raw_input("Press return to quit.")

    for i in range(5):
        #pygame.event.get()
        length = np.random.randint(0, 50)
        start = np.random.randint(0, 1593-length)
        ledIDs = list(range(start, start + length))
        col = np.random.randint(0, 128, size=3)
        cols = np.ones((length, 3))*col
        dis.setLeds(ledIDs, cols)
        dis.window.clock.tick(30)

    test_results.record("setLeds()", True)
    raw_input("Press return to quit.")

    for i in range(5):
        #pygame.event.get()
        length = leds.numCells
        ledIDs = list(range(length))
        col = np.random.randint(0, 128, size=3)
        cols = np.ones((length, 3))*col
        dis.setAllLeds(cols)
        dis.window.clock.tick(30)

    test_results.record("setAllLeds()", True)
    raw_input("Press return to quit.")

    dis.show_image("images/monalisa.png")
    test_results.record("show_image()", True)

    print("Testing complete.")
    raw_input("Press return to quit.")

    dis.save_state()


if __name__ == "__main__":

    run_tests()

    logging.info("Testing complete.")