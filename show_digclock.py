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
for the tick and call `dis.show()` - so the one call that actually
makes the display update happens with nothing else between it and the
tick, and a minute rollover's digit change and flash toggle land in the
same hardware refresh instead of two separate ones.

Colour
------
By default the digits are plain red. With `--temp-colour`, they are
coloured by the current outdoor temperature, in 10-degree bands: cyan
below -10 C, then blue, green, red, and yellow from 20 C
(`T_COLOUR_BANDS`, `temperature_colour`). An
`OutdoorTemperature` background thread polls Environment Canada's
citypage weather API, so a slow or failed request never delays a tick.
Observations are hourly, so it polls `WEATHER_POLL_OFFSET_SECS` after
the next one is due (from the current one's observation time), then
every `WEATHER_RETRY_SECS` until a newer observation time appears. The
colour is only re-evaluated at a minute rollover, when digits are being
staged anyway; if it changed, every lit LED is re-staged. Until a
temperature has been fetched successfully the digits are red; after
that, a failed update just keeps the last known value.

Brightness
----------
The digit colour's fully lit PWM values come from
`display1593.colour_correction`, so every digit colour looks equally
bright, at the hourly brightness setting in `BCYCLE` (or a fixed one
with `--brightness`). Each LED's value from the face (rendered at full
intensity, `FACE_BNESS`) is its coverage by a digit segment, and scales
those PWM values linearly, so partly lit edge LEDs keep the digits'
shape. (Putting the face values through the colour correction curve
instead made the digits look thin, with gaps between segments.)
"""

import argparse
import json
import logging
import threading
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from display1593 import Display1593
from display1593.colour_correction import (
    apply_colour_lut,
    apply_coverage,
    colour_lut,
)
from display1593.digclock1 import SevenSegmentClockFace
from display1593.logging_utils import configure_root_logging

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
LOG_PATH = BASE_DIR / "show_digclock.log"
N_LEDS = 1593

configure_root_logging(LOG_PATH)

# Environment Canada citypage weather: current conditions for Calgary
# (location ab-52, observed at Calgary Int'l Airport).
WEATHER_URL = (
    "https://api.weather.gc.ca/collections/citypageweather-realtime"
    "/items/ab-52?f=json"
)
# Observations are hourly: poll this long after the next one is due,
# then every WEATHER_RETRY_SECS until it appears (or after a failure).
WEATHER_OBS_INTERVAL = timedelta(hours=1)
WEATHER_POLL_OFFSET_SECS = 60
WEATHER_RETRY_SECS = 5 * 60
WEATHER_TIMEOUT_SECS = 10

# Temperature colour bands: (upper limit in deg C, RGB colour), in
# increasing order; temperatures at or above the last limit get
# T_COLOUR_ABOVE. A continuous blue-magenta-red scale was tried first,
# but its colours (e.g. magenta) didn't look good and changes of a few
# degrees were too small to notice. Colour components are 0 or 1, so
# each lit channel gets the same requested intensity, which the colour
# correction makes equally bright for every colour.
T_COLOUR_BANDS = [
    (-10.0, np.array([0.0, 1.0, 1.0])),  # cyan
    (0.0, np.array([0.0, 0.0, 1.0])),  # blue
    (10.0, np.array([0.0, 1.0, 0.0])),  # green
    (20.0, np.array([1.0, 0.0, 0.0])),  # red
]
T_COLOUR_ABOVE = np.array([1.0, 1.0, 0.0])  # yellow
# Digit colour by default, and with --temp-colour before any temperature
# has been fetched.
DEFAULT_COLOUR = np.array([1.0, 0.0, 0.0])

# Brightness setting (display1593.colour_correction) by hour of day:
# the day/night dimming cycle. Settings 0-7 are the calibrated range;
# the night settings are below it. Chosen (2026-11) so the brightest
# red digit LEDs are about as bright as with the earlier cycle of PWM
# divisors (2, 3, 5, 8, 9 -> settings 7, 5, 1, -2, -3: peak red PWM
# 110, 83, 48, 31, 27 vs. 127, 85, 51, 31, 28); not yet tuned by eye.
BCYCLE = {
    0: -3,
    1: -3,
    2: -3,
    3: -3,
    4: -3,
    5: -3,
    6: -2,
    7: 1,
    8: 5,
    9: 7,
    10: 7,
    11: 7,
    12: 7,
    13: 7,
    14: 5,
    15: 5,
    16: 5,
    17: 1,
    18: 1,
    19: -2,
    20: -3,
    21: -3,
    22: -3,
    23: -3,
}

# The clock face renders at full intensity (brightness divisor 1); the
# dimming comes from the colour correction brightness setting.
FACE_BNESS = 1


def paint(smem, idx, vals):
    """
    Add (idx, vals) - as returned by a clock face's digit_leds()/
    flash_leds() - into smem, summing where a physical LED is shared by
    more than one digit position or by a digit and the flash element.
    """
    if idx.size:
        total = smem[idx].astype(np.uint16) + vals
        smem[idx] = np.minimum(total, 255).astype(smem.dtype)


def clear_digit(smem, clear_idx):
    """Zero out all LEDs belonging to one digit position."""
    if clear_idx.size:
        smem[clear_idx] = 0


def push_changes(dis, smem, smem_prev, initialized, colour, lut=None):
    """
    Stage any LEDs whose value changed since the last push (or that have
    never been pushed at all).

    This only transfers the new values to the microcontrollers (set_leds);
    it does NOT call dis.show(). Staging is the slow part (a serial
    write per changed LED), so callers do it ahead of a tick and trigger
    the actual display update with a bare show() right after
    SecondTicker.wait_for_tick() returns, keeping that gap as small as
    possible.

    Each LED's RGB value is its smem brightness times `colour` (three
    floats, 0-1), passed through the colour correction table `lut` if
    given (see colour_rgb).
    """
    changed = np.nonzero((smem != smem_prev) | ~initialized)[0]
    if changed.size > 0:
        rgb_array = colour_rgb(smem[changed], colour, lut)
        dis.set_leds(changed, rgb_array)
        smem_prev[changed] = smem[changed]
        initialized[changed] = True


class SecondTicker:
    """
    Tracks wall-clock seconds and lets callers wait for the next tick.

    Centralizing this here means every part of the script that needs to
    know "has the next second arrived yet" advances off the same clock,
    instead of each sampling datetime.now() independently.
    """

    def __init__(self):
        self.time = datetime.now().time()

    def wait_for_tick(self, spin=0.005):
        """
        Wait until the wall-clock second changes, then return the new time.

        Sleeps until `spin` seconds before the next second, then checks
        the time in a tight loop for the last few milliseconds, so the tick
        is still caught within microseconds (callers show() right after
        it) while using almost no CPU. A pure busy-wait kept one core at
        100% all the time. If the next second is less than `spin` away, or
        has already passed, it doesn't sleep.
        """
        now = datetime.now()
        to_next = 1.0 - now.microsecond / 1e6
        if now.second == self.time.second and to_next > spin:
            time.sleep(to_next - spin)
        while datetime.now().time().second == self.time.second:
            pass
        self.time = datetime.now().time()
        return self.time


class OutdoorTemperature:
    """
    Polls the current outdoor temperature (deg C) from Environment
    Canada in a daemon thread. `value` is the last temperature fetched
    successfully, or None if there hasn't been one yet, and `observed`
    its observation time (UTC datetime).

    The thread waits before its first poll, so call update() once
    before start() to get a value straight away.
    """

    def __init__(self, url=WEATHER_URL):
        self.url = url
        self.value = None
        self.observed = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def update(self):
        """Fetch the temperature; on failure, log and keep the old value."""
        try:
            with urllib.request.urlopen(
                self.url, timeout=WEATHER_TIMEOUT_SECS
            ) as response:
                props = json.load(response)["properties"]
            conditions = props["currentConditions"]
            temp = float(conditions["temperature"]["value"]["en"])
            # e.g. "2026-10-03T16:00:00Z" (fromisoformat needs +00:00
            # for "Z" before Python 3.11).
            observed = datetime.fromisoformat(
                conditions["timestamp"]["en"].replace("Z", "+00:00")
            )
        except Exception as e:
            logger.warning("Outdoor temperature update failed: %r", e)
            return
        if observed != self.observed:
            logger.info(
                "Outdoor temperature %.1f C (observed %s)", temp, observed
            )
        self.value = temp
        self.observed = observed

    def seconds_until_next_poll(self):
        """
        Time until WEATHER_POLL_OFFSET_SECS after the next observation is
        due, or WEATHER_RETRY_SECS if that's already passed (or there's
        no observation yet).
        """
        if self.observed is None:
            return WEATHER_RETRY_SECS
        due = self.observed + WEATHER_OBS_INTERVAL
        wait = (due - datetime.now(timezone.utc)).total_seconds()
        wait += WEATHER_POLL_OFFSET_SECS
        return wait if wait > 0 else WEATHER_RETRY_SECS

    def _run(self):
        while not self._stop.wait(self.seconds_until_next_poll()):
            self.update()


def temperature_colour(temp):
    """
    Digit colour (three floats, 0-1) for an outdoor temperature in deg C,
    from T_COLOUR_BANDS, or DEFAULT_COLOUR if temp is None.
    """
    if temp is None:
        return DEFAULT_COLOUR
    for limit, colour in T_COLOUR_BANDS:
        if temp < limit:
            return colour
    return T_COLOUR_ABOVE


def colour_rgb(vals, colour, lut=None):
    """
    (n, 3) uint8 PWM values for LEDs with face brightness values `vals`
    (0-255, a digit segment's coverage of each LED) in `colour` (three
    floats, 0-1): the colour's fully lit PWM values, from the colour
    correction table `lut` (colour_correction.colour_lut; without one,
    `colour` * 255), scaled linearly by vals / 255. Coverage is linear
    so partly covered edge LEDs keep the digits' shape (see
    colour_correction's docstring).
    """
    colour_255 = np.rint(np.asarray(colour) * 255).astype("uint8")
    peak = colour_255 if lut is None else apply_colour_lut(colour_255, lut)
    return apply_coverage(peak, np.asarray(vals) / 255)


def flash_rgb(vals, second, colour, lut=None):
    """
    RGB values for the flash element for a given wall-clock second
    (on/off toggle).
    """
    return colour_rgb((second % 2) * vals, colour, lut)


def hour_digits(hr):
    """Return (tens, ones) digit values for an hour, 0-23."""
    return hr // 10, hr % 10


def minute_digits(m):
    """Return (tens, ones) digit values for a minute, 0-59."""
    return m // 10, m % 10


def hour_lut(hr, brightness=None):
    """
    Colour correction table for hour `hr`, at brightness setting
    `brightness` if given (fixed for every hour), otherwise BCYCLE's
    setting for the hour.
    """
    if brightness is None:
        brightness = BCYCLE[hr % 24]
    return colour_lut(brightness)


def main(temp_colour=False, brightness=None):
    face = SevenSegmentClockFace()

    smem = np.zeros(N_LEDS, dtype="uint8")
    smem_prev = np.zeros(N_LEDS, dtype="uint8")
    initialized = np.zeros(N_LEDS, dtype=bool)

    weather = None
    colour = DEFAULT_COLOUR
    if temp_colour:
        weather = OutdoorTemperature()
        # First fetch before starting, so the first frame is coloured.
        weather.update()
        weather.start()
        colour = temperature_colour(weather.value)

    with Display1593() as dis:
        ticker = SecondTicker()
        t = ticker.wait_for_tick()
        hr, m = t.hour, t.minute
        d4, d3 = hour_digits(hr)
        d2, d1 = minute_digits(m)

        lut = hour_lut(hr, brightness)

        # Initial paint of all four digits (positions 0-3, left to right).
        for position, value in enumerate((d4, d3, d2, d1)):
            paint(smem, *face.digit_leds(position, value, FACE_BNESS))

        flash_idx, flash_vals = face.flash_leds(FACE_BNESS)

        # First frame is a special case: there's no earlier tick to stage
        # ahead of, since we needed *this* tick to know what to paint.
        push_changes(dis, smem, smem_prev, initialized, colour, lut)
        dis.show()
        logger.info("%2d:%2d", hr, m)

        while True:
            next_second = (t.second + 1) % 60

            if next_second == 0:
                m = (m + 1) % 60
                if m == 0:
                    hr = (hr + 1) % 24
                    # Update brightness before any digit is repainted, so
                    # the minute digits (repainted below) also get the new
                    # hour's brightness rather than keeping the old one
                    # until they next change.
                    lut = hour_lut(hr, brightness)
                    # The brightness is in the table, not smem, so
                    # re-stage every lit LED.
                    initialized[smem > 0] = False

                d1 = m % 10
                clear_digit(smem, face.clear_indices(3))
                paint(smem, *face.digit_leds(3, d1, FACE_BNESS))

                if d1 == 0:
                    d2 = m // 10
                    clear_digit(smem, face.clear_indices(2))
                    paint(smem, *face.digit_leds(2, d2, FACE_BNESS))

                if m == 0:
                    d4, d3 = hour_digits(hr)

                    clear_digit(smem, face.clear_indices(1))
                    paint(smem, *face.digit_leds(1, d3, FACE_BNESS))

                    clear_digit(smem, face.clear_indices(0))
                    paint(smem, *face.digit_leds(0, d4, FACE_BNESS))

                # If the outdoor temperature changed the colour, re-stage
                # every lit LED, not just the ones whose brightness changed.
                if weather is not None:
                    new_colour = temperature_colour(weather.value)
                    if not np.array_equal(new_colour, colour):
                        colour = new_colour
                        initialized[smem > 0] = False

                # Stage the new digits now, ahead of the tick that will
                # make them current.
                push_changes(dis, smem, smem_prev, initialized, colour, lut)

            # Stage the flash element's state for the second we're about
            # to enter, then wait for it to actually arrive before showing
            # anything - a minute rollover's digit change and flash toggle
            # land in the same show().
            dis.set_leds(
                flash_idx, flash_rgb(flash_vals, next_second, colour, lut)
            )

            t = ticker.wait_for_tick()
            dis.show()

            if next_second == 0:
                logger.info("%2d:%2d", hr, m)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--temp-colour",
        action="store_true",
        help="colour the digits by the current outdoor temperature "
        "(Calgary, from Environment Canada) instead of plain red",
    )
    parser.add_argument(
        "--brightness",
        type=float,
        default=None,
        help="fixed brightness setting for every hour, instead of the "
        "BCYCLE day/night cycle (0-7 is the calibrated range; below 0 is "
        "dimmer)",
    )
    args = parser.parse_args()
    logger.info("=" * 35)
    logger.info("%s started.", __file__)
    main(
        temp_colour=args.temp_colour,
        brightness=args.brightness,
    )
