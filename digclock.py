#!/usr/bin/python
"""------------------- digclock.py -------------------
Python module to display the time as a digital clock
similar to an old LCD alarm clock. Uses the Irregular
1593-LED display built by me.

Author: Bill Tubbs
This was verion last updated: December 2017

Execute the module directly to display the current
time as follows:

$ python digclock.py

Or import and use the dig_clock module in python:

>>> import digclock
>>> dclock = digclock.DigitalClock(color='r')
>>> dclock.show_time_string("09:45")

Note, digclock.py requires the following components:
- display1593
"""

import logging
import pickle
import sys
from datetime import datetime
import display1593 as display


logging.basicConfig(
	filename='logfile.txt',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


# Load LED intensity data to display the shapes that represent
# all the segments of each LCD digit (x4) plus the two dots
# that look like a ':' symbol.
# The complete display looks something like this: '88:88'
with open('digclock/digdata.pickle', 'rb') as handle:
    dig_data = pickle.load(handle)

# Data for the 4 digits are stored in:
#  dig_data[1]
#  dig_data[2]
#  dig_data[3]
#  dig_data[4]
# Data for the ':' symbol is in dig_data[0]

# The following dictionary stores how to represent each number
# (0-9) with the 7 LCD segments of each digit display.
# Key 10 is for the two dots (':' symbol).
d_chars = {
        0: [0, 1, 2, 4, 5, 6],
        1: [5, 6],
        2: [1, 2, 3, 4, 5],
        3: [2, 3, 4, 5, 6],
        4: [0, 3, 5, 6],
        5: [0, 2, 3, 4, 6],
        6: [0, 1, 2, 3, 4, 6],
        7: [2, 5, 6],
        8: [0, 1, 2, 3, 4, 5, 6],
        9: [0, 2, 3, 4, 5, 6],
        10: [7]
        }

# brightness values - use these to vary the display
# brightness over a 24-hour period
bcycle = {
    0: 9,
    1: 7,
    2: 7,
    3: 7,
    4: 7,
    5: 7,
    6: 6,
    7: 4,
    8: 2,
    9: 1,
    10: 1,
    11: 1,
    12: 1,
    13: 1,
    14: 1,
    15: 1,
    16: 1,
    17: 2,
    18: 3,
    19: 5,
    20: 7,
    21: 7,
    22: 7,
    23: 7
}

class DigitalClock(object):
    """DigitalClock class"""

    def __init__(self, color='r', imax=64):
        """Create an instance of DigitalClock.

        Arguments:
        color  - string containing only 'r', 'g' and 'b'.
        imax   - maximum intensity value (brightness) of LEDs."
        """

        if not all([c in 'rgb' for c in color]):
            raise ValueError("color must be a string containing only 'r', 'g' "
                             "and 'b'.")
        self.rgb = [c in color for c in 'rgb']
        self.imax = imax

        # Connect to LED display
        self.dis = display.Display1593()
        self.dis.connect()
        self.dis.clear()

        # Initialise display memory
        self.smem = [0]*1593
        self. smem_prev = [0]*1593

    def display_time(self, time_string):
        """Display the time provided in time_string.

        time_string must be 5 characters in length and be in
        the format 'HH:MM' where HH are the hour digits and MM
        are the minute digits.  Spaces are allowed and the
        colon may be omitted."""

        self.clear_all()

        digits = [time_string[i] for i in [0, 1, 3, 4]]
        for i, d in enumerate(digits):
            if d not in "01234567890 ":
                raise ValueError("time_string must contain only digits (0-9) "
                                 "or spaces")
            if d != " ":
                self.add_digit(i+1, int(d))

        dots = time_string[2]
        if dots not in " :":
            raise ValueError("time_string must contain a ':' or space between the 2nd and 3rd digits.")
        if dots == ':':
            self.add_dots()

        self.update_display()

    def add_dots(self):
        """Add colour intensities for the ':' symbol."""

        for n in range(2):
            for i, x in dig_data[0][n].items():
                self.smem[i] += x * self.imax // 255

    def clear_dots(self):
        """Set colour intensities for the ':' symbol
        to zero."""

        for n in range(2):
            for i, x in dig_data[0][n].items():
                self.smem[i] = 0

    def add_digit(self, d, n):
        """Add colour intensities for the number n to
        digit d."""

        for s in d_chars[n]:
            for i, x in dig_data[d][s].items():
                self.smem[i] += x * self.imax // 255

    def clear_digit(self, d):
        """Add colour intensities for the number n to
        digit d."""

        for s in d_chars[8]:
            for i, x in dig_data[d][s].items():
                self.smem[i] = 0

    def update_display(self):
        """Update the LED display with the changes made since
        last update."""

        for i in range(1593):
            if self.smem[i] != self.smem_prev[i]:
                self.dis.setLed(i, tuple(c*self.smem[i] for c in self.rgb))
                self.smem_prev[i] = self.smem[i]

    def clear_all(self):
        """Clear all digits and dots."""

        self.clear_dots()
        for d in range(1, 5):
            self.clear_digit(d)

def main(argv):
    """Demonstration of how to use this module."""

    logging.info("\n\n------------- Digital Clock Display -------------\n")
    logging.info("Displays current time as a digital clock similar to an ")
    logging.info("old LCD alarm clock.")

    # Get current time
    t = datetime.now()
    hr, min = (t.hour, t.minute)
    d4, d3 = (hr / 10), (hr % 10)
    d2, d1 = (min / 10), (min % 10)

    logging.info("Displaying clock...")

    try:
        color = argv[0]
    except IndexError:
        color = 'r'

    try:
        imax = int(argv[1])
    except IndexError:
        imax = 48

    dclock = DigitalClock(color=color, imax=imax)

    time_string = datetime.strftime(t, "%H:%M")
    dclock.display_time(time_string)

    hr, min, s = t.hour, t.minute, t.second

    while True:
        logging.info("%02d:%02d" % (hr, min))

        while t.minute == min:
            while datetime.now().time().second == s:
                pass
            if (s % 2) == 0:
                dclock.add_dots()
            else:
                dclock.clear_dots()
            t = datetime.now().time()
            s = t.second
            dclock.update_display()

        min = t.minute
        if min == 0:
            hr = t.hour

        # Set value for digit 1
        d1 = (min % 10)
        dclock.clear_digit(4)
        dclock.add_digit(4, d1)

        if d1 == 0:

            # Set value for digit 2
            d2 = (min // 10)

            dclock.clear_digit(3)
            dclock.add_digit(3, d2)

        if min == 0:

            # Set values for digits 3 and 4
            d4, d3 = (hr // 10), (hr % 10)

            dclock.clear_digit(2)
            dclock.add_digit(2, d3)

            dclock.clear_digit(1)
            dclock.add_digit(1, d4)

if __name__ == "__main__":
    main(sys.argv[1:])
