# Python code for RaspberryPi to communicate with
# Teensy driving the LED display
#
# See here:
# http://blog.oscarliang.net/connect-raspberry-pi-and-arduino-usb-cable/

import display1593 as display
from datetime import datetime
import numpy as np
from scipy import ndimage

# To identify new devices connected to Raspberry Pi USB ports
# Use type lsusb into the shell
# Example:
# Bus 001 Device 006: ID 16c0:0483 VOTI Teensyduino Serial
# Bus 001 Device 005: ID 16c0:0483 VOTI Teensyduino Serial

# To find the port name of each Teensy use the following with
# and without the Teensies plugged in:
# ls /dev/tty*


bcycle = {
    0: 120,
    1: 120,
    2: 120,
    3: 120,
    4: 120,
    5: 120,
    6: 120,
    7: 100,
    8: 50,
    9: 25,
    10: 18,
    11: 16,
    12: 16,
    13: 16,
    14: 16,
    15: 16,
    16: 18,
    17: 25,
    18: 50,
    19: 100,
    20: 120,
    21: 120,
    22: 120,
    23: 120
}

# --------------------- START OF MAIN FUNCTION ---------------------


def main():

    print "--------------------- show_images.py ---------------------"

    dis = display.Display1593()
    dis.connect()

    dis.clear()

    fnames = [
        "images/monalisa.png",
        "images/bowie.jpg",
        #"einstein.jpeg"
        #"muhammedali.jpg",
        "images/spock.jpg",
        "images/hendrix.jpg"
    ]

    i = 0
    status = ""
    col = 0

    while True:

        for f in fnames:

            # Get current time
            start_time = datetime.now()
            hr, mn, sc = (start_time.hour, start_time.minute, start_time.second)

            # Set display dimmer level
            bness = bcycle[hr % 24]

            print "Hour", hr, "Brightness:", bness

            dis.show_image(f)

            while (datetime.now() - start_time).total_seconds() < 60*10:
                pass

    exit()



# ------------------------- END OF MAIN CODE ------------------------


if __name__ == '__main__':  main()

