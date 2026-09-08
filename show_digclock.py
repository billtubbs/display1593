"""
Digital clock display driver for a 1593-LED physical display.

This script owns the generic, display-face-agnostic parts of running the
clock: connecting to the hardware, timing (`SecondTicker`), the day/night
brightness cycle (`BCYCLE`), compositing digit and flash intensities into
`smem`, and pushing only the changed LEDs to the display.

Rendering - deciding which physical LEDs light up, at what intensity, to
represent a given digit value or the once-per-second flash element - is
delegated to a "clock face" object (currently `SevenSegmentClockFace`,
defined in `display1593.digclock1`, the first such face). Any clock face
is expected to provide: a fixed number of decimal digit positions
(`n_digits`), `digit_leds(position, value, bness)`,
`clear_indices(position)`, and `flash_leds(bness)` - see
`display1593.digclock1` for the concrete implementation and the pickle
data format it reads. Swapping in a different rendering style (e.g. a
dot-matrix face, in a `digclock2` module) means writing an alternative
class with that same interface; nothing else in this script needs to
change.

Program flow
------------
`smem` holds the current target brightness (0-255) for every one of the
1593 LEDs. `smem_prev` holds the values last actually pushed to the
display hardware, paired with an `initialized` boolean mask so every LED
gets sent at least once on the very first pass, without needing a sentinel
value in `smem_prev` itself. Where a physical LED is shared by more than
one digit position (or by a digit and the flash element), `paint()` sums
their contributions into `smem` rather than overwriting.

On startup, all digit positions are painted into `smem` once and shown.
The main loop then runs once per wall-clock second, driven by a
`SecondTicker`: on each iteration it stages (but does not yet display) the
flash element's state for the upcoming second, and, when the upcoming
second is about to roll the minute over, also stages repainted digits for
the new time. Only after all of that staging is done does the loop wait
for the tick and call `dis.show_now()` - so the one call that actually
makes the display update happens with nothing else between it and the
tick, and a minute rollover's digit change and flash toggle land in the
same hardware refresh instead of two separate ones.
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from display1593 import Display1593
from display1593.logging_utils import configure_root_logging
from display1593.digclock1 import SevenSegmentClockFace

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
LOG_PATH = BASE_DIR / "show_digclock.log"
N_LEDS = 1593

configure_root_logging(LOG_PATH)

# Brightness divisor by hour-of-day (day/night dimming cycle).
BCYCLE = {
    0: 9,
    1: 9,
    2: 9,
    3: 9,
    4: 9,
    5: 9,
    6: 8,
    7: 5,
    8: 3,
    9: 1,
    10: 1,
    11: 1,
    12: 2,
    13: 2,
    14: 3,
    15: 3,
    16: 3,
    17: 5,
    18: 5,
    19: 8,
    20: 9,
    21: 9,
    22: 9,
    23: 9,
}


def paint(smem, idx, vals):
    """
    Add (idx, vals) - as returned by a clock face's digit_leds()/
    flash_leds() - into smem, summing where a physical LED is shared by
    more than one digit position or by a digit and the flash element.
    """
    if idx.size:
        smem[idx] += vals.astype(smem.dtype)


def clear_digit(smem, clear_idx):
    """Zero out all LEDs belonging to one digit position."""
    if clear_idx.size:
        smem[clear_idx] = 0


def push_changes(dis, smem, smem_prev, initialized):
    """
    Stage any LEDs whose value changed since the last push (or that have
    never been pushed at all).

    This only transfers the new values to the microcontrollers (set_leds);
    it does NOT call dis.show_now(). Staging is the slow part (a serial
    write per changed LED), so callers do it ahead of a tick and trigger
    the actual display update with a bare show_now() right after
    SecondTicker.wait_for_tick() returns, keeping that gap as small as
    possible.
    """
    changed = np.nonzero((smem != smem_prev) | ~initialized)[0]
    if changed.size > 0:
        rgb_array = np.zeros((changed.size, 3), dtype="uint8")
        rgb_array[:, 0] = smem[changed]
        dis.set_leds(changed, rgb_array)
        smem_prev[changed] = smem[changed]
        initialized[changed] = True


class SecondTicker:
    """
    Tracks wall-clock seconds and lets callers busy-wait for the next tick.

    Centralizing this here means every part of the script that needs to
    know "has the next second arrived yet" advances off the same clock,
    instead of each sampling datetime.now() independently.
    """

    def __init__(self):
        self.time = datetime.now().time()

    def wait_for_tick(self):
        """Busy-wait until the wall-clock second changes, then return the new time."""
        while datetime.now().time().second == self.time.second:
            pass
        self.time = datetime.now().time()
        return self.time


def flash_rgb(vals, second):
    """RGB values for the flash element for a given wall-clock second (on/off toggle)."""
    rgb = np.zeros((vals.size, 3), dtype="uint8")
    rgb[:, 0] = (second % 2) * vals
    return rgb


def hour_digits(hr):
    """Return (tens, ones) digit values for an hour, 0-23."""
    return hr // 10, hr % 10


def minute_digits(m):
    """Return (tens, ones) digit values for a minute, 0-59."""
    return m // 10, m % 10


def main():
    face = SevenSegmentClockFace()

    smem = np.zeros(N_LEDS, dtype="uint8")
    smem_prev = np.zeros(N_LEDS, dtype="uint8")
    initialized = np.zeros(N_LEDS, dtype=bool)

    with Display1593() as dis:
        ticker = SecondTicker()
        t = ticker.wait_for_tick()
        hr, m = t.hour, t.minute
        d4, d3 = hour_digits(hr)
        d2, d1 = minute_digits(m)

        bness = BCYCLE[hr % 24]

        # Initial paint of all four digits (positions 0-3, left to right).
        for position, value in enumerate((d4, d3, d2, d1)):
            paint(smem, *face.digit_leds(position, value, bness))

        flash_idx, flash_vals = face.flash_leds(bness)

        # First frame is a special case: there's no earlier tick to stage
        # ahead of, since we needed *this* tick to know what to paint.
        push_changes(dis, smem, smem_prev, initialized)
        dis.show_now()
        logger.info("%2d:%2d", hr, m)

        while True:
            next_second = (t.second + 1) % 60

            if next_second == 0:
                m = (m + 1) % 60
                if m == 0:
                    hr = (hr + 1) % 24

                d1 = m % 10
                clear_digit(smem, face.clear_indices(3))
                paint(smem, *face.digit_leds(3, d1, bness))

                if d1 == 0:
                    d2 = m // 10
                    clear_digit(smem, face.clear_indices(2))
                    paint(smem, *face.digit_leds(2, d2, bness))

                if m == 0:
                    bness = BCYCLE[hr % 24]
                    d4, d3 = hour_digits(hr)

                    clear_digit(smem, face.clear_indices(1))
                    paint(smem, *face.digit_leds(1, d3, bness))

                    clear_digit(smem, face.clear_indices(0))
                    paint(smem, *face.digit_leds(0, d4, bness))

                    flash_idx, flash_vals = face.flash_leds(bness)

                # Stage the new digits now, ahead of the tick that will
                # make them current.
                push_changes(dis, smem, smem_prev, initialized)

            # Stage the flash element's state for the second we're about
            # to enter, then wait for it to actually arrive before showing
            # anything - a minute rollover's digit change and flash toggle
            # land in the same show_now().
            dis.set_leds(flash_idx, flash_rgb(flash_vals, next_second))

            t = ticker.wait_for_tick()
            dis.show_now()

            if next_second == 0:
                logger.info("%2d:%2d", hr, m)


if __name__ == "__main__":
    logger.info("=" * 35)
    logger.info("%s started.", __file__)
    main()
