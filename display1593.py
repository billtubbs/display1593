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
from timeit import default_timer as timer
from scipy import ndimage
from PIL import Image

# If using Python 2:
#from itertools import izip

logging.basicConfig(
    filename='logfile.txt',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Load irregular LED array data
import ledArray_data_1593 as leds

with open('data/mask1593b.pickle', 'rb') as f:
    mask1593 = pickle.load(f)   # load object from file

# Load RGB colour intensity calibration scales
rgb_scales = np.load('data/rgb_scales.npy')

# Connection details

# These are specific to the computer you are running
# this code on

# To identify new devices connected to the Raspberry Pi's
# USB ports type lsusb -v into the shell

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
    """Teensy micro-controller connection object class."""

    def __init__(self, address, baud=9600):

        self.address = address
        self.baud = baud
        self.serial = None
        self.name = None

    def connect(self, timeout=2):

        # Establish connection
        self.serial = serial.Serial(self.address, self.baud,
                                    timeout=timeout)
        #self.serial.open()

        # Request identification
        response = self.get_id()

        result = False
        if response is not None:
            if response.startswith(b"Teensy"):
                self.name = response.decode("utf-8")
                logging.info("Connection to %s (%s) successful.",
                             self.address, response)
                result = True

        return result

    def get_id(self):

        # Request identification
        self.serial.write(b'ID')

        try:
            response = self.serial.readline().rstrip()
        except:
            logging.info("Connection to %s failed.", self.address)
            self.serial.close()
            return None

        return response

    def check_connection(self):

        result = False
        try:
            response = self.get_id()
        except serial.SerialException:
            pass
        else:
            if response.decode("utf-8") == self.name:
                logging.info("Connection to %s (%s) checked.",
                             self.name, self.address)
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
        for filename in os.listdir('/dev'):
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

    def check_connections(self):

        return all([tys.check_connection() for tys in self.tys])

    def setLed(self, led, col):
        """Set colours of an individual LED."""

        ledRef = leds.ledIndex[led]
        i = ledRef[1]

        # Send bytes b'S', LED #, col, b'\n'
        data = bytes((83, i >> 8, i % 256, col[0], col[1], col[2], 10))
        self.tys[ledRef[0]].serial.write(data)
        #logging.info("Led %d set to 0x%2x%2x%2x [t%d: %d]", led, col[0],
        #             col[1], col[2], ledRef[0], i)

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

        # prepare two lists to accumulate colour values destined
        # for each controller
        s = [list(), list()]
        cnt = [0, 0]

        for i, col in zip(ledIDs, cols):
            controller = leds.ledIndex[i][0]
            ledRef = leds.ledIndex[i][1]
            s[controller].append(ledRef >> 8)
            s[controller].append(ledRef % 256)
            for c in range(3):
                s[controller].append(col[c])
            cnt[controller] += 1

        for i in range(leds.numberOfControllers):
            n = len(s[i])
            if n > 0:
                # Send b'N' + LED # + colours
                data = bytes([78, n >> 8, n % 256] + [item for item in s[i]])
                self.tys[i].serial.write(data)

            logging.info("%d leds set on Teensy %d", cnt[i], i)

    @staticmethod
    def send_bytes(controller, data):

        n = controller.serial.write(data)

        if n != len(data):
            raise Display1593Error(
                  "Error sending data to Teensy. "
                  "{} bytes sent out of {}.".format(n, len(data))
                  )

    def setAllLeds(self, cols):
        """setAllLeds(self, cols)
        Set all LEDs to the specified sequence of colours.
        Args:
            cols: sequence of 1593 (r, g, b) values or an
                array of values of shape (1593, 3).
        """
        # TODO: Still getting occasional data errors
        # with the following LEDs: 42, 840.

        if len(cols) != leds.numCells:
            raise Display1593Error("setAllLeds() requires a sequence of"
                                   " %d colours." % leds.numCells)

        data = np.array(cols, dtype='int8')
        # Set dud LEDs to black (doesn't fix problem)
        data[42] = 0
        data[840] = 0
        data = data.tobytes()

        mid_point = leds.numLeds[0]*3
        self.send_bytes(self.tys[0], b'A' + data[0:mid_point])
        self.send_bytes(self.tys[1], b'A' + data[mid_point:])

    def comm_test(self, show=True):

        logging.info("Starting communication test...")

        # TODO: Use part of nearest neighbour data for test data
        # data = (leds.nearestNeighbours[:, 0] % 256).astype('int8').tobytes()
        # mid_point = leds.numLeds[0]
        data = bytes(range(1, 251))

        responses = []
        timings = []
        for tys in self.tys:
            start = timer()
            self.send_bytes(tys, b'X' + data + b'\n')
            response = tys.serial.readline().rstrip()
            end = timer()
            responses.append(response)
            timings.append(end - start)

        if show:
            for i, x in enumerate(zip(responses, timings)):
                print("Teensy %d: %s, %.4fs" %
                      (i, x[0].decode('utf8'), x[1]))

        if all([r == b'OK' for r in responses]):
            logging.info("Communication test successful.")
            return True
        else:
            logging.info("Communication test failed.")
            return False

    def clear(self):
        """Set all LEDs to black (0, 0, 0)."""

        for controller in self.tys:
            controller.serial.write(b'CLS')
        logging.info("Display cleared")

    def getBrightness(self):
        """getBrightness() -> int

        Gets and returns the analog reading from the
        photo-resistor connected to Teensy 1. This value
        should be an integer in the range 0 to 1023.
        """

        for controller in self.tys:
            if controller.name == 'Teensy1':
                controller.serial.write(b'B')
                data = controller.serial.read(size=2)
                return int.from_bytes(data, 'big')

    def getColour(self, led):
        """getColour(led) -> (int, int, int)

        Get the colour of an individual LED."""

        led_ref = leds.ledIndex[led]
        controller = self.tys[led_ref[0]]
        i = led_ref[1]

        # Send b'G' + LED #
        data = b'G' + i.to_bytes(2, byteorder='big')
        controller.serial.write(data)
        col = controller.serial.read(size=3)

        return tuple(col)

    def prepare_image(self, image, size=(256, 256)):
        """Adjusts the size of image and returns a 256x256
        pixel image suitable for use with the method
        convert_image()."""

        if isinstance(image, Image.Image):
            image = np.array(image)

        logging.info("Preparing image %s", str(image.shape))

        # Find out how many colour layers the image has
        image_shape = image.shape[:2]
        if len(image.shape) > 2:
            n_cols = image.shape[2]
        else:
            n_cols = 1

        data_type = image.dtype

        # Reduce redundant layers - if imageCols > 1,
        # assume first three are red, green, blue and
        # ignore the rest

        if n_cols > 3:
            image = image[:, :, :3]
            n_cols = image.shape[2]
            logging.info("Image reduced to %d layers (R, G, B)", n_cols)

        # Determine smallest dimension (x or y)
        image_size = min(image_shape)

        if not image_shape[0] == image_shape[1]:

            if image_shape[0] > image_size:
                logging.info("Image is not square. Y-axis will be cropped.")
                y = (image_shape[0] - image_size)//2
                image = image[y:y+image_size, :, :]

            elif image_shape[1] > image_size:
                logging.info("Image is not square. X-axis will be cropped.")
                x = (image_shape[1] - image_size)//2
                image = image[:, x:x+image_size, :]

            logging.info("Cropped image size: %dx%d", image.shape[0],
                         image.shape[1])

        # Adjust maximum colour intensities of final
        # image

        max_int = np.amax(image, axis=(0, 1)).tolist()
        if n_cols > 1:
            logging.info("Maximum RGB values: %s", str(max_int))

        else:
            logging.info("Maximum B&W image intensity: %d", max_int)
            image = np.stack([image]*3, axis=2)
        # Re-size image to required pixel size for super-sampling
        # algorithm

        # Re-size image ready for super-sampling
        # Replace with
        s_image = np.array(Image.fromarray(image).resize(size))

        logging.info("Image re-sized: %s", str(s_image.shape[:2]))

        return s_image

    def convert_image(self, image_array):
        """Convert 256 x 256 RGB image array to 1593 RGB led intensities."""

        global mask1593

        # Flatten the first 2 dimensions (x, y)
        img_data = image_array.reshape(-1, *image_array.shape[2:])

        return np.mean(img_data[mask1593], axis=1).astype(int)

    def show_image(self, filename, dimness=8):

        img = ndimage.imread(filename)
        z = self.convert_image(self.prepare_image(img))
        self.setAllLeds(z**2/(256*dimness))

    def show_image_calibrated(self, img, resize=True,
                              brightness=2,
                              rgb_scales=rgb_scales):

        if brightness == 0:
            self.clear()

        else:
            if isinstance(img, str):
                img = ndimage.imread(img)

            if resize:
                z = self.convert_image(self.prepare_image(img))
            else:
                z = self.convert_image(img)

            # Uses rgb_scales array to calibrate intensities
            rgb = np.array([0, 1, 2]*leds.numCells)
            z = rgb_scales[brightness - 1][
                (z.ravel()//8, rgb)
            ].reshape(leds.numCells, 3)

            self.setAllLeds(z)

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

    @staticmethod
    def image_snapshot(image, point, size):
        x = point[0]
        y = point[1]
        width = size[0]
        height = size[1]
        return image[y-height/2:y+height/2, x-width/2:x+width/2, :]

    def __repr__(self):

        return "Display1593()"

