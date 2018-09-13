#!/usr/bin/python
# ---------- Irregular LED Display driver module ----------
# This module written in Python contains class definitions
# and methods to control an irregular array of 1593 LEDs
# designed and built by Bill Tubbs in 2014-2015.

# The code is intended to run on a supervisor processor
# such as a Raspberry Pi connected to the display.  The
# display itself is driven by two Teensy microcontrollers.
# The processor running this code communicates with the
# Teensies using a serial connection.

# Teensy communication code list

# 'ID' - Send identification message in response
# 'S' - Set the colour of one LED
# 'N' - Update a batch of n LED colours
# 'A' - Update all LED colours
# 'G' - Get the colour of an LED and return it
# 'CLS' - Clear screen (to black)

import serial
import time
import os
import pickle
import logging
import numpy as np
from scipy import ndimage
from scipy import misc
from itertools import izip

logging.basicConfig(
	filename='logfile.txt',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Load irregular LED array data
import ledArray_data_1593 as leds

f = open('/home/pi/code/data/mask1593.pickle', 'r')
mask1593 = pickle.load(f)   # load object from file
f.close()


# Connection details

# These are specific to the computer you are running
# this code on

# To identify new devices connected to the Raspberry Pi's
# USB ports type lsusb into the shell

# Example:
# Bus 001 Device 006: ID 16c0:0483 VOTI Teensyduino Serial
# Bus 001 Device 005: ID 16c0:0483 VOTI Teensyduino Serial

# To find the port name of each Teensy use the following with
# and without the Teensies plugged in:
# ls /dev/tty*

addresses = ('/dev/ttyACM0', '/dev/ttyACM1')

class Display1593Error(Exception):
    """Exception class for reporting display1593 errors."""

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class Teensy(object):

    def __init__(self, address, baud=38400):

        self.address = address
        self.baud = baud
        self.serial = None
        self.name = None

    def connect(self):

        # Establish connection
        self.serial = serial.Serial(self.address, self.baud)
        #self.serial.open()

        # Request identification
        self.serial.write('ID\n')

        result = False
        try:
            response = self.serial.readline().rstrip()
        except:
            logging.info("Connection to %s failed.", self.address)
            self.serial.close()
            return result

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

    def __init__(self):
        self.tys = None

    def connect(self, addresses=addresses):

        logging.info("Finding Teensies...")

        tty_files = []
        for file in os.listdir('/dev'):
            if file.startswith('ttyACM'):
                tty_files.append(file)

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

        ledRef = leds.ledIndex[led]
        id = ledRef[1]
        self.tys[ledRef[0]].serial.write('S' + chr(id >> 8) + chr(id % 256) \
            + chr(col[0]) + chr(col[1]) + chr(col[2]) +'\n')
        #logging.info("Led %d set to 0x%2x%2x%2x [t%d: %d]", led, col[0],
        #             col[1], col[2], ledRef[0], id)

    def setLeds(self, ledIDs, cols):
        """setLeds(self, ledIDs, cols)

        Set colours of a batch of LEDs.

        Arguments
        ---------

        ledIDs  - list or tuple containing the LED ID numbers
        cols    - list or tuple containing the LED colour
                  intensities (r, g, b).
        """

        if len(cols) != len(ledIDs):
            raise Display1593Error("ledIDs and cols must be sequences"
                                   " of equal length.")
            return False

        # prepare two lists to accumulate colour values destined
        # for each controller
        s = [list(), list()]
        cnt = [0, 0]

        for id, col in izip(ledIDs, cols):
            controller = leds.ledIndex[id][0]
            ledRef = leds.ledIndex[id][1]
            s[controller].append(chr(ledRef >> 8))
            s[controller].append(chr(ledRef % 256))
            for c in range(3):
                s[controller].append(chr(col[c]))
            cnt[controller] += 1

        for i in range(leds.numberOfControllers):
            n = len(s[i])
            if n>0:
                self.tys[i].serial.write('N' + chr(n >> 8) + chr(n % 256))
                self.tys[i].serial.write("".join([item for item in s[i]]))

            logging.info("%d leds set on Teensy %d", cnt[i], i)

    def setLeds(self, ledIDs, cols):
        """setLeds(self, ledIDs, cols)
        Set colours of a batch of LEDs.
        Arguments
        ---------
        ledIDs  - list or tuple containing the LED ID numbers
        cols    - list or tuple containing the LED colour
                  intensities (r, g, b).
        """

        if len(cols) != len(ledIDs):
            raise Display1593Error("ledIDs and cols must be sequences"
                                   " of equal length.")
            return False

        # prepare two lists to accumulate colour values destined
        # for each controller
        s = [list(), list()]
        cnt = [0, 0]

        for id, col in izip(ledIDs, cols):
            controller = leds.ledIndex[id][0]
            ledRef = leds.ledIndex[id][1]
            s[controller].append(chr(ledRef >> 8))
            s[controller].append(chr(ledRef % 256))
            for c in range(3):
                s[controller].append(chr(col[c]))
            cnt[controller] += 1

        for i in range(leds.numberOfControllers):
            n = len(s[i])
            if n>0:
                self.tys[i].serial.write('N' + chr(n >> 8) + chr(n % 256))
                self.tys[i].serial.write("".join([item for item in s[i]]))

            logging.info("%d leds set on Teensy %d", cnt[i], i)

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

        start = 0
        for i, controller in enumerate(self.tys):
            end = start + leds.numLeds[i]
            char_data = 'A' + "".join(["{:c}{:c}{:c}".format( \
                        c[0], c[1], c[2]) for c in cols[start:end]])
            self.send_chars(controller, char_data)
            start = end

    def setAllLeds2(self, cols):
        """DO NOT USE - NOT WORKING
        Set all LEDs to the specified sequence of colours.

        Args:
            cols: sequence of 1593 (r, g, b) values or an
                array of values of shape (1593, 3).
        """
        # TODO: This is faster than setAllLeds but causes a communication
        # breakdown with Teensy.  Then can't reconnect...

        if len(cols) != leds.numCells:
            raise Display1593Error("setAllLeds() requires a sequence of"
                                   " %d colours." % leds.numCells)

        char_data = np.array(cols, dtype='int8').ravel() \
                        .view('S{}'.format(cols.size))[0]

        mid_point = leds.numLeds[0]*3
        self.send_chars(self.tys[0], 'A{}'.format(char_data[0:mid_point]))
        self.send_chars(self.tys[1], 'A{}'.format(char_data[mid_point:]))

    def send_chars(self, controller, char_data):

        n = controller.serial.write(char_data)

        if n != len(char_data):
            raise Display1593Error(
                  "Error sending data to Teensy. " \
                  "{} bytes sent out of {}.".format(n, len(char_data))
                  )

    def clear(self):
        """Set all LEDs to black (0, 0, 0)."""

        for controller in self.tys:
            controller.serial.write('CLS')

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

        ledRef = leds.ledIndex[led]
        controller = self.tys[ledRef[0]]
        id = ledRef[1]
        controller.serial.write('G' + chr(id >> 8) + chr(id % 256))
        col = controller.serial.read(size=3)
        return (ord(col[0]), ord(col[1]), ord(col[2]))

    def prepare_image(self, image):
        """Adjusts the size of the ndimage stored in image and
        returns a 256x256 pixel image suitable for use with
        the method convert_image()."""

        #logging.info("Preparing image %s", str(image.shape))

        # Find out how many layers the image has
        image_shape = image.shape[:2]
        if len(image.shape)>2:
            imageCols = image.shape[2]
        else:
            imageCols = 1

        data_type = image.dtype

        # Reduce redundant layers - if imageCols > 1,
        # assume first three are red, green, blue and
        # ignore the rest

        if imageCols > 3:
            image = image[:,:,:3]
            imageCols = image.shape[2]
            #logging.info("Image reduced to %d layers (R, G, B)", imageCols)

        # Determine smallest dimension (x or y)
        imageSize = min(image_shape)

        if not image_shape[0] == image_shape[1]:
            if image_shape[0] > imageSize:
                #logging.info("Image is not square. Y-axis will be cropped.")
                y = (image_shape[0] - imageSize)/2
                image = image[:][y:y+imageSize]
            elif image_shape[1] > imageSize:
                #logging.info("Image is not square. X-axis will be cropped.")
                x = (image_shape[1] - imageSize)/2
                image = image[:, x:x+imageSize, :]

            imageShape = image.shape[:2]
            #logging.info("Cropped image size: %dx%d", image_shape[0],
            #             image_shape[1])

        # Define arrays for maximum colour intensities
        # of final image

        cols = ['red', 'green', 'blue']

        if imageCols > 1:
            maxInt = np.zeros((imageCols), dtype = 'uint8')
            for i in range(imageCols):
                selCol = np.array([i], dtype=np.intp)
                colMin = image[:, :, selCol].min()
                colMax = image[:, :, selCol].max()
                #logging.info("Colour %d range: %d to %d", i, colMin, colMax)
                maxInt[i] = max(255,colMax)
        else:
            logging.info("This is a black and white image.")
            logging.info("Intensity range: %d to %d", image.min(), image.max())
            logging.info("Specify RGB intensities for final image (0-31):")

            maxInt = np.zeros((3), dtype = 'uint8')
            for i, c in enumerate(cols):
                while True:
                    s = raw_input(c + ": ")
                    try:
                        f = float(s)
                    except:
                        print("Enter a number between 0 and 255")
                        continue
                    maxInt[i] = f
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

        with picamera.PiCamera() as camera:
            camera.resolution = (1024, 768)
            camera.start_preview()
            # Camera warm-up time
            time.sleep(2)
            camera.capture(filename)

        logging.info("Image saved to file", filename)

    def image_snapshot(self, image, point, size):
        x = point[0]
        y = point[1]
        width = size[0]
        height = size[1]
        return image[y-height/2:y+height/2, x-width/2:x+width/2, :]

    def __repr__(self):

        return "Display1593()"

