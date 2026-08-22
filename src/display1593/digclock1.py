"""
Seven-segment digital clock face rendering.

Interprets the physical LED layout described by a pickled digit/segment/LED
map into per-digit and per-flash LED intensities, for a clock face made of
four decimal digit positions (tens/ones of hours, tens/ones of minutes,
left to right) plus one "flash" element (e.g. colon dots) that a driver
can blink once per second.

Pickle data format
------------------
The pickle is a dict keyed by position number (0-4):

    0  - flash element (e.g. colon dots)
    1  - digit 4 (leftmost, tens of hours)
    2  - digit 3 (ones of hours)
    3  - digit 2 (tens of minutes)
    4  - digit 1 (rightmost, ones of minutes)

Each position's value is itself a dict mapping a 7-segment segment number
to a further dict `{led_index: raw_brightness_value, ...}`. A single
segment is made up of several individual LEDs, each with its own raw
brightness value, because physical LEDs vary and need individually-tuned
values to make the lit segment look evenly bright. Digit positions use
segments 0-6 (the classic 7-segment layout); the flash position uses
segments 0-1 for its two dot LEDs and has no digit shape to encode.
"""

import logging
import pickle
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_PICKLE_PATH = _DATA_DIR / "digdata.pickle"

_EMPTY_IDX = np.empty(0, dtype=np.int64)
_EMPTY_VALS = np.empty(0, dtype=np.int64)

# Which segments (0-6) are lit to display each digit value, 0-9.
_D_CHARS = {
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
}

# All segment numbers that make up a digit's shape.
_DIGIT_SEGMENTS = range(7)

# The two dot LEDs at the flash position use segments 0-1.
_FLASH_SEGMENTS = range(2)

N_DIGITS = 4


class SevenSegmentClockFace:
    """
    A 4-digit, 7-segment-style clock face plus a once-per-second flash
    element, backed by a pickled LED map for one physical display.

    Digit positions are numbered 0-3, left to right (0 = leftmost). The
    flash element (position 0 in the pickle) is addressed separately via
    `flash_leds()`, since it has no digit value to render.
    """

    n_digits = N_DIGITS

    def __init__(self, path=DEFAULT_PICKLE_PATH):
        dig_data = self._load(path)
        self._positions = self._preprocess(dig_data)
        self._clear_idx = [
            self._clear_indices_for(p) for p in self._positions[1:]
        ]
        self._flash_idx, self._flash_vals = self._flash_leds_for(
            self._positions[0]
        )
        self._check_positions_disjoint()

    @staticmethod
    def _load(path):
        """Load the raw digit/segment/LED mapping data from disk."""
        with open(path, "rb") as handle:
            dig_data = pickle.load(handle)
        logger.info("data for %d digit segments unpickled.", len(dig_data))
        return dig_data

    @staticmethod
    def _preprocess(dig_data):
        """
        Convert the raw dict-of-dicts pickle structure into numpy arrays for
        fast vectorized indexing.

        Returns a list of dicts (one per position: flash, then digits 0-3),
        each mapping a segment number to an (idx_array, val_array) pair of
        matching length.
        """
        if not isinstance(dig_data, dict):
            raise TypeError("dig_data must be a dict keyed by position")

        processed = []
        for key in sorted(dig_data):
            position_data = dig_data[key]
            if not isinstance(position_data, dict):
                raise TypeError(
                    f"position {key} must be a dict of segment mappings"
                )

            segments = {}
            for n, seg in position_data.items():
                if not isinstance(seg, dict):
                    raise TypeError(
                        f"segment {n} in position {key} must be a dict"
                    )
                idx = np.fromiter(seg.keys(), dtype=np.int64)
                vals = np.fromiter(seg.values(), dtype=np.int64)
                segments[n] = (idx, vals)
            processed.append(segments)
        return processed

    @staticmethod
    def _clear_indices_for(position_data):
        idx_arrays = [
            position_data[n][0]
            for n in _DIGIT_SEGMENTS
            if n in position_data and position_data[n][0].size
        ]
        if not idx_arrays:
            return _EMPTY_IDX
        # A boundary LED can belong to more than one of the digit's
        # segments; np.unique collapses that down to one zeroing write.
        return np.unique(np.concatenate(idx_arrays))

    @staticmethod
    def _flash_leds_for(position_data):
        idx_arrays = [
            position_data[n][0]
            for n in _FLASH_SEGMENTS
            if n in position_data and position_data[n][0].size
        ]
        if not idx_arrays:
            return _EMPTY_IDX, _EMPTY_VALS
        val_arrays = [
            position_data[n][1]
            for n in _FLASH_SEGMENTS
            if n in position_data and position_data[n][1].size
        ]
        idx = np.concatenate(idx_arrays)
        vals = np.concatenate(val_arrays)
        return _sum_duplicate_leds(idx, vals)

    def _check_positions_disjoint(self):
        """
        Verify no physical LED belongs to more than one digit position (or
        to both a digit and the flash element).

        The driver composes positions independently - clear_digit() zeroes
        one digit's LEDs at a time, and paint() sums each position's
        contribution into smem separately - which is only correct if
        positions never share LEDs. A shared LED would mean clearing one
        digit could wipe out part of another, still-lit one.
        """
        labelled = [("flash", self._flash_idx)]
        labelled += [
            (f"digit {i}", idx) for i, idx in enumerate(self._clear_idx)
        ]
        seen = {}
        for label, idx in labelled:
            for led in idx.tolist():
                if led in seen:
                    raise ValueError(
                        f"LED {led} belongs to both {seen[led]!r} and "
                        f"{label!r}; SevenSegmentClockFace assumes every "
                        "position (digits and flash) occupies disjoint "
                        "LEDs, since clear_digit()/paint() clear and "
                        "repaint one position at a time."
                    )
                seen[led] = label

    def clear_indices(self, position):
        """LED indices to zero out before repainting `position` (0-3, left to right)."""
        return self._clear_idx[position]

    def digit_leds(self, position, value, bness):
        """
        (idx, vals) to light `value` (0-9) at `position` (0-3, left to
        right), scaled by the brightness divisor `bness`.

        A physical LED sitting on the boundary between two of the digit's
        own segments appears in both segments' raw data; its two
        contributions are summed rather than one silently overwriting the
        other.
        """
        position_data = self._positions[position + 1]
        idx_arrays = []
        val_arrays = []
        for n in _D_CHARS[value]:
            idx, vals = position_data.get(n, (_EMPTY_IDX, _EMPTY_VALS))
            if idx.size:
                idx_arrays.append(idx)
                val_arrays.append(vals // bness)
        if not idx_arrays:
            return _EMPTY_IDX, _EMPTY_VALS
        idx = np.concatenate(idx_arrays)
        vals = np.concatenate(val_arrays)
        return _sum_duplicate_leds(idx, vals)

    def flash_leds(self, bness):
        """(idx, vals) to light the flash element, scaled by `bness`."""
        return self._flash_idx, self._flash_vals // bness


def _sum_duplicate_leds(idx, vals):
    """
    Collapse repeated LED indices into unique ones, summing their values.

    Needed because a single physical LED can appear in more than one
    segment's raw data (e.g. at a corner shared by two adjacent
    segments); a plain `smem[idx] += vals` with duplicate indices in one
    call does NOT accumulate them (numpy keeps only one), so duplicates
    must be summed before that point.
    """
    unique_idx, inverse = np.unique(idx, return_inverse=True)
    if unique_idx.size == idx.size:
        return idx, vals
    summed = np.zeros(unique_idx.size, dtype=vals.dtype)
    np.add.at(summed, inverse, vals)
    return unique_idx, summed
