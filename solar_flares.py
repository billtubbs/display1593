#!/usr/bin/python
"""Python script to download image of solar flare from
https://www.spaceweatherlive.com and display a close-up
image of the brightest current flare on the 1593 LED
display.
"""

import os
import sys
import logging
import time
from datetime import datetime
import requests
from io import BytesIO
import numpy as np
from scipy import ndimage
from scipy.signal import convolve2d
from PIL import Image
import display1593 as display
import argparse


parser = argparse.ArgumentParser(description='Display solar flares')
parser.add_argument('-d', '--delay', type=int, default=15,
                    help='Image refresh rate (mins)')
args = parser.parse_args()


def download_image(url, headers=None):
    """Download image from internet http address and return
    PIL image.
    """
    response = requests.get(url, headers=headers)
    return Image.open(BytesIO(response.content))


def find_brightest_area(img, crop_size, filter_size=None):
    """Finds the brightest region of the image and returns
    the co-ordinates of the box that contains the brightest
    part at its centre.
    """

    # Convert to 2d array by summing colour channels
    img_array = np.array(img).mean(axis=2)

    x_size, y_size = crop_size
    if filter_size is None:
        # Too large filter takes long time to compute
        filter_size = min(x_size, 32), min(y_size, 32)

    # Filter is a rectangular matrix
    f = np.ones(filter_size)
    f = f / f.size

    convolution = convolve2d(img_array, f, mode="same")
    assert convolution.shape == img_array.shape

    max_level = convolution.max()
    y, x = np.where(convolution == max_level)
    assert (len(y) == 1) & (len(x) == 1)

    # Use first result if more than one maximum
    # TODO: could pick one in viscinity of most others
    y, x = int(y[0]), int(x[0])

    left, top, right, bottom = (
        x - x_size // 2,
        y - y_size // 2,
        x + x_size // 2,
        y + y_size // 2
    )

    return left, top, right, bottom


# Image of the sun "right now" from https://www.spaceweatherlive.com
url = "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_0131.jpg"

images_dir = 'images'
filename1 = 'solar_image.jpg'
filename2 = 'solar_flare.jpg'

logging.basicConfig(
	filename='logfile.txt',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Image brightness is based on photo resistor readings
brightness_levels = {
    0: 4.5,
    1: 12.2323,
    2: 33.2508,
    3: 90.3849,
    4: 245.692,
    5: 667.859,
    6: 1815.43,
    7: 4934.85,
    8: 13414.3,
    9: 36463.9,
    10: 99119.1,
    11: 269434
}
# Convert to numpy array for easier indexing
brightness_levels = np.array(list(brightness_levels.items()))


def calculate_brightness_level(brightness):
    """Converts a reading from the photo-resistor (typically
    in the range 2.2 to 160.0) into an integer brightness
    level for use with the show_image_calibrated method of
    Display1593 instances.
    """
    if brightness > brightness_levels[:, 1].max():
        level = brightness_levels[:, 0].max()
    else:
        idx = np.argmax(brightness_levels[:, 1] > brightness)
        level = brightness_levels[idx, 0]
    return int(level)


def main():

    logging.info("\n\n----------------- solar_flares.py -----------------\n")

    logging.info("Image refresh rate (mins): %d", args.delay)
    crop_size = (128, 128)
    logging.info("Snapshot image size: %s", crop_size)

    dis = display.Display1593()
    dis.connect()
    dis.clear()

    status = "load"
    load_time = None

    # Parameter for calculating exponentially-weighted
    # moving average (EWMA)
    alpha = 0.1

    # Initialize with average of series of readings from sensor
    brightness_ewma = np.array(
        [dis.getBrightness() for i in range(5)]
    ).mean()
    brightness_level = 0  # Image will be displayed first time

    while True:

        if status == 'load':
            # Timer to count down to next image refresh
            load_time = time.time()
            logging.info("Downloading new image...")
            headers = {
                'User-Agent': 'solar_flares.py',
                'From': 'bill.tubbs@me.com'
            }
            img = download_image(url, headers=headers)
            logging.info("New image downloaded")
            logging.info("Image size: %s", img.size)
            img_mb = sys.getsizeof(img.tobytes()) // 1000000
            logging.info("Image size (MB): %.2f", img_mb)
            filepath = os.path.join(images_dir, filename1)
            img.save(filepath)
            #img = Image.open(filepath)

            # Trim image to remove text
            margin = 48
            img_cropped = img.crop(
                (margin, margin, img.size[0] - margin, img.size[1] - margin)
            )
            logging.info("Finding brightest spot...")
            left, top, right, bottom = find_brightest_area(
                img_cropped,
                crop_size=crop_size
            )
            logging.info("Co-ordinates: (%d, %d, %d, %d)", left, top,
                         right, bottom)
            img_snapshot = img_cropped.crop((left, top, right, bottom))
            logging.info("Snapshot size: %s", img_snapshot.size)
            filepath = os.path.join(images_dir, filename2)
            img_snapshot.save(filepath)
            logging.info("Snapshot image saved to '%s'", filepath)
            status = 'display'
        elif status == 'wait':
            if (time.time() - load_time) // 60 > args.delay:
                status = 'load'

        # Get current time
        start_time = datetime.now()
        sec = start_time.second

        # Read light level from photo resistor
        brightness = dis.getBrightness()

        # Update EWMA
        brightness_ewma = alpha * brightness + (1 - alpha) * brightness_ewma

        # See if image needs updating
        new_level = calculate_brightness_level(brightness_ewma)
        if new_level != brightness_level or status == 'display':
            if new_level != brightness_level:
                brightness_level = new_level
                logging.info("Brightness: %d", brightness_level)
            filepath = os.path.join(images_dir, filename2)
            dis.show_image_calibrated(filepath, brightness=brightness_level)
            logging.info("Image displayed: '%s'", filename2)
            if status == 'display':
                status = 'wait'

        while (datetime.now().second == sec):
            time.sleep(0.10)  # Short wait


if __name__ == '__main__':
    main()
