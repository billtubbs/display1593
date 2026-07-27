"""
Digital clock display driver for a 1593-LED physical display.

The display is built from individual LEDs arranged into five 7-segment-style
"digit positions" plus a pair of colon dots, physically split across two
Teensy-driven boards. The `Display1593` class abstracts that split away, so
this script only needs to think in terms of a single flat array of 1593 LED
brightness values.

Pickle data format (digdata.pickle)
------------------------------------
`dig_data` is a list of 5 elements, one per digit position:

    dig_data[0]  - decimal point / colon dots
    dig_data[1]  - digit 4 (leftmost, tens of hours)
    dig_data[2]  - digit 3 (ones of hours)
    dig_data[3]  - digit 2 (tens of minutes)
    dig_data[4]  - digit 1 (rightmost, ones of minutes)

Each `dig_data[position]` is itself a dict mapping a 7-segment segment number
(0-6, using the standard 7-segment layout; segment/key 7 is used for the two
colon dot "segments" in position 0) to a further dict:

    {led_index: raw_brightness_value, ...}

A single "segment" of a digit is made up of several individual LEDs, each
with its own raw brightness value, because physical LEDs vary and need
individually-tuned values to make the lit segment look evenly bright.

`d_chars` maps a digit value (0-9) to the list of segment numbers that must
be lit to display that digit, using the classic 7-segment encoding.

`bcycle` maps hour-of-day (0-23) to a brightness divisor ("bness"), giving
the display a day/night dimming cycle - dimmer late at night, brighter
during the day.

Program flow
------------
`smem` holds the current target brightness (0-255) for every one of the
1593 LEDs. `smem_prev` holds the values last actually pushed to the
display hardware, paired with an `initialized` boolean mask so every LED
gets sent at least once on the very first pass, without needing a sentinel
value in `smem_prev` itself.

On startup, the hour digits and the tens-of-minutes digit are painted into
`smem` once. Then, in the main loop: the ones-of-minutes digit is painted,
any changed LEDs are pushed to the display, and the colon dots are flashed
once per second until the minute changes. At that point the ones-of-minutes
digit (and, when relevant, the higher digit positions) are cleared and
repainted for the new time.
"""

import logging
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s.%(msecs)03d|%(levelname)s|%(name)s|%(message)s",
)

from display1593 import Display1593

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
PICKLE_PATH = BASE_DIR / "digdata.pickle"
N_LEDS = 1593

# Which segments (0-6) are lit to display each digit value, 0-9.
# Key 10 (segment 7) is used by the colon dots, not a digit value.
D_CHARS = {
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
    10: [7],
}

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
    14: 2,
    15: 2,
    16: 2,
    17: 3,
    18: 5,
    19: 8,
    20: 9,
    21: 9,
    22: 9,
    23: 9,
}

# Segments 0-6 are cleared when repainting a digit position; segment 7
# (colon dots, position 0 only) is handled separately by flash_dots().
CLEARABLE_SEGMENTS = range(7)

_EMPTY_IDX = np.empty(0, dtype=np.int64)
_EMPTY_VALS = np.empty(0, dtype=np.int64)


def load_dig_data(path=PICKLE_PATH):
    """Load the raw digit/segment/LED mapping data from disk."""
    with open(path, "rb") as handle:
        dig_data = pickle.load(handle)
    logger.info("data for %d digit segments unpickled.", len(dig_data))
    return dig_data


def preprocess_dig_data(dig_data):
    """
    Convert the raw dict-of-dicts pickle structure into numpy arrays for
    fast vectorized indexing.

    Returns a list of 5 dicts (one per digit position), each mapping a
    segment number to an (idx_array, val_array) pair of matching length,
    ready for direct use as `smem[idx_array] = val_array // bness`.
    """
    processed = []
    for position_data in dig_data:
        segments = {}
        for n, seg in position_data.items():
            idx = np.fromiter(seg.keys(), dtype=np.int64)
            vals = np.fromiter(seg.values(), dtype=np.int64)
            segments[n] = (idx, vals)
        processed.append(segments)
    return processed


def digit_clear_indices(position_data):
    """All LED indices used by segments 0-6 of one digit position (for clearing)."""
    idx_arrays = [
        position_data[n][0]
        for n in CLEARABLE_SEGMENTS
        if n in position_data and position_data[n][0].size
    ]
    if not idx_arrays:
        return _EMPTY_IDX
    return np.concatenate(idx_arrays)


def apply_segments(smem, position_data, segments, bness, accumulate=False):
    """
    Light (or accumulate into) the LEDs for the given segment numbers of
    one digit position, scaled by the current brightness divisor.
    """
    for n in segments:
        idx, vals = position_data.get(n, (_EMPTY_IDX, _EMPTY_VALS))
        if idx.size == 0:
            continue
        scaled = (vals // bness).astype(smem.dtype)
        if accumulate:
            smem[idx] += scaled
        else:
            smem[idx] = scaled


def clear_digit(smem, clear_idx):
    """Zero out all LEDs belonging to one digit position."""
    if clear_idx.size:
        smem[clear_idx] = 0


def push_changes(dis, smem, smem_prev, initialized):
    """
    Send only the LEDs whose value changed since the last push (or that
    have never been pushed at all).
    """
    changed = np.nonzero((smem != smem_prev) | ~initialized)[0]
    if changed.size > 0:
        rgb_array = np.zeros((changed.size, 3), dtype="uint8")
        rgb_array[:, 0] = smem[changed]
        dis.set_leds(changed, rgb_array)
        smem_prev[changed] = smem[changed]
        initialized[changed] = True
        dis.show_now()


def flash_dots(dis, points_idx, points_vals, bness, m):
    """Flash the colon dots once per second until the minute changes."""
    t = datetime.now().time()
    s = t.second
    dot_rgb = np.zeros((points_idx.size, 3), dtype="uint8")

    while t.minute == m:
        while datetime.now().time().second == s:
            pass
        dot_rgb[:, 0] = (s % 2) * (points_vals // bness)
        dis.set_leds(points_idx, dot_rgb)
        dis.show_now()
        t = datetime.now().time()
        s = t.second


def hour_digits(hr):
    """Return (tens, ones) digit values for an hour, 0-23."""
    return hr // 10, hr % 10


def minute_digits(m):
    """Return (tens, ones) digit values for a minute, 0-59."""
    return m // 10, m % 10


def main():
    dig_data = load_dig_data()
    processed = preprocess_dig_data(dig_data)
    clear_idx = [digit_clear_indices(p) for p in processed]

    smem = np.zeros(N_LEDS, dtype="uint8")
    smem_prev = np.zeros(N_LEDS, dtype="uint8")
    initialized = np.zeros(N_LEDS, dtype=bool)

    with Display1593() as dis:
        t = datetime.now().time()
        hr, m = t.hour, t.minute
        d4, d3 = hour_digits(hr)
        d2, d1 = minute_digits(m)

        bness = BCYCLE[hr % 24]

        # Initial paint of points, hours, and tens-of-minutes.
        # (smem starts at 0, so accumulate vs. overwrite is equivalent here.)
        apply_segments(smem, processed[0], range(2), bness, accumulate=True)
        apply_segments(smem, processed[1], D_CHARS[d4], bness, accumulate=True)
        apply_segments(smem, processed[2], D_CHARS[d3], bness, accumulate=True)
        apply_segments(smem, processed[3], D_CHARS[d2], bness, accumulate=True)

        # Colon dot LEDs/values, precomputed once for the flashing loop.
        points_idx = np.concatenate(
            [processed[0][n][0] for n in range(2) if processed[0][n][0].size]
        )
        points_vals = np.concatenate(
            [processed[0][n][1] for n in range(2) if processed[0][n][1].size]
        )

        while True:
            apply_segments(
                smem, processed[4], D_CHARS[d1], bness, accumulate=True
            )
            push_changes(dis, smem, smem_prev, initialized)

            t = datetime.now().time()
            hr, m = t.hour, t.minute
            logger.info("%2d:%2d", hr, m)

            flash_dots(dis, points_idx, points_vals, bness, m)

            m = (m + 1) % 60
            if m == 0:
                hr = (hr + 1) % 24

            d1 = m % 10

            if d1 == 0:
                d2 = m // 10
                clear_digit(smem, clear_idx[3])
                apply_segments(
                    smem, processed[3], D_CHARS[d2], bness, accumulate=True
                )

            if m == 0:
                bness = BCYCLE[hr % 24]
                d4, d3 = hour_digits(hr)

                clear_digit(smem, clear_idx[2])
                apply_segments(
                    smem, processed[2], D_CHARS[d3], bness, accumulate=True
                )

                clear_digit(smem, clear_idx[1])
                apply_segments(
                    smem, processed[1], D_CHARS[d4], bness, accumulate=True
                )

            clear_digit(smem, clear_idx[4])


if __name__ == "__main__":
    logger.info("=" * 35)
    logger.info("%s started.", __file__)
    main()
