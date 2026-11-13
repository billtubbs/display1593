"""
Colour balance and intensity correction for the display's LEDs.

The LEDs' perceived brightness is far from linear in their 0-255 PWM
values, and the three colours aren't equally bright at the same value.
This module turns a requested intensity per channel (0-255, meaning
"perceived" intensity: equal steps look equally spaced, and equal
values look equally bright across channels) into the PWM values to send,
at one of the display's brightness settings.

The correction comes from a by-eye calibration on the display itself,
with `diagnostics/colour_balance_test.py` (first done in 2016, re-checked
in November 2026): per channel, an exponential scale of N_LEVELS levels,
base**x - 1 for x from 0 to x_max (SCALE_PARAMS), set equal across
channels at the lowest levels so dim colours don't change hue, then
multiplied by per-channel gains (DEFAULT_GAINS) and by
BRIGHTNESS_STEP**brightness.

Brightness settings 0-7 are the calibrated range, with peak PWM values
of about 41-110 (red), 49-131 (green), 80-214 (blue); the display is for
indoor use, so it never needs anything near full PWM. Other settings
(including negative and fractional ones) extrapolate the same scales,
e.g. for a dimmer display at night.

Usage, in any display script:

    lut = colour_lut(brightness=3)
    dis.set_all_leds(apply_colour_lut(rgb, lut))   # rgb: (n, 3), 0-255

Scripts that don't want the correction just send PWM values directly.
Kept import-light (numpy only) so it can be used without the hardware
dependencies.
"""

import numpy as np

N_LEVELS = 32
CHANNELS = "rgb"

# Per-channel scale constants (base, x_max): level i of channel c is
# base**x - 1, with x = linspace(0, x_max, N_LEVELS)[i]. From the 2016
# calibration.
SCALE_PARAMS = {
    "r": (2.5, 4.2),
    "g": (2.72, 3.9),
    "b": (2.85, 4.2),
}

# Low levels are set to the mean of the three channels (same grey for
# all) while that mean is <= LOW_LEVEL_MEAN and level <= LOW_LEVEL_MAX.
LOW_LEVEL_MEAN = 2.5
LOW_LEVEL_MAX = 10

# Default per-channel gains, applied after the low-level equalisation.
# In a 2026-11 retest of the 2016 scales, red's brightest levels looked
# slightly brighter than green and blue (the rest looked balanced); a
# red gain of 0.9-0.95 fixed it, with 0.9 better at brightness 6-7.
DEFAULT_GAINS = {"r": 0.9, "g": 1.0, "b": 1.0}

# Brightness setting n multiplies the scales by BRIGHTNESS_STEP**n;
# settings 0 to N_BRIGHTNESS - 1 are the calibrated range.
N_BRIGHTNESS = 8
BRIGHTNESS_STEP = 1.15


def make_rgb_scales(gains=None, brightness=0):
    """
    (N_LEVELS, 3) float array of PWM values for each level and channel,
    at brightness setting `brightness`, with per-channel `gains` (three
    factors, default DEFAULT_GAINS) applied after the low-level
    equalisation.
    """
    if gains is None:
        gains = [DEFAULT_GAINS[ch] for ch in CHANNELS]
    scales = np.zeros((N_LEVELS, 3))
    for c, ch in enumerate(CHANNELS):
        base, x_max = SCALE_PARAMS[ch]
        scales[:, c] = base ** np.linspace(0, x_max, N_LEVELS) - 1

    # Make the channels identical at low intensity levels.
    for i, rgb in enumerate(scales):
        avg = rgb.mean()
        if avg > LOW_LEVEL_MEAN or i > LOW_LEVEL_MAX:
            break
        scales[i, :] = avg

    return scales * np.asarray(gains) * BRIGHTNESS_STEP**brightness


def colour_lut(brightness=0, gains=None):
    """
    (256, 3) uint8 look-up table from requested intensity (0-255) to PWM
    value, per channel, at brightness setting `brightness`. Intensity v
    maps to level v / 255 * (N_LEVELS - 1) of make_rgb_scales(),
    interpolating linearly between levels. Values above 255 (only at
    brightness settings well above the calibrated range) are clipped.
    """
    scales = make_rgb_scales(gains, brightness)
    x = np.arange(256) / 255 * (N_LEVELS - 1)
    levels = np.arange(N_LEVELS)
    lut = np.column_stack(
        [np.interp(x, levels, scales[:, c]) for c in range(3)]
    )
    return np.clip(np.rint(lut), 0, 255).astype("uint8")


def apply_colour_lut(rgb, lut):
    """
    PWM values (uint8, same shape) for an array of requested intensities
    `rgb`, shape (..., 3), integers 0-255, using a table from
    colour_lut().
    """
    rgb = np.asarray(rgb)
    return lut[rgb.astype(np.intp), np.arange(3)]
