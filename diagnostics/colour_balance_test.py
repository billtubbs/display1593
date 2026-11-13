#!/usr/bin/env python3
"""
Interactive colour balance and intensity linearity test.

Port of the 2016 `coltest.py` (display1593-old repo). Builds an
intensity scale per colour channel (N_LEVELS levels, each of the form
base**x - 1 with x from 0 to X_MAX), equalised across channels at the
lowest levels so dim colours don't shift hue, and displays it as a
chart: 8 vertical bars (white, black, red, yellow, green, cyan, blue,
magenta), each with N_LEVELS rows from dimmest (top) to brightest
(bottom). Adjust the scales until the colours look equally bright
across each row and the steps down each bar look even.

The 2016 calibration (saved as data/rgb_scales.npy in the old repo)
came out with green about 1.05x red and blue about 1.7x red, but only
covered PWM values up to about 46-80. show_digclock.py's digits go up
to 127 in the daytime (BCYCLE divisor 2) and down to 28 at night
(divisor 9), so the brightness settings here span roughly that range.

Commands (typed at the prompt, then Enter):
  0-7          brightness setting (peak red value ~ 46 * 1.15**n)
  r|g|b GAIN   set a channel's gain (default DEFAULT_GAINS), e.g. "g 0.8"
  s            save the current scale to a CSV file
  q            quit (clears the display)

Run from the repo root, on the Pi:
    python diagnostics/colour_balance_test.py
    python diagnostics/colour_balance_test.py --dry-run   # no hardware
"""

import argparse

import numpy as np

N_LEVELS = 32

# Per-channel scale constants (base, x_max): level i of channel c is
# base**x - 1, with x = linspace(0, x_max, N_LEVELS)[i]. Values from
# the 2016 coltest.py.
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

# Brightness setting n multiplies the scales by BRIGHTNESS_STEP**n.
N_BRIGHTNESS = 8
BRIGHTNESS_STEP = 1.15
DEFAULT_BRIGHTNESS = 2

# Bar colours, left to right.
COL_BARS = np.array(
    [
        (1, 1, 1),  # white
        (0, 0, 0),  # black
        (1, 0, 0),  # red
        (1, 1, 0),  # yellow
        (0, 1, 0),  # green
        (0, 1, 1),  # cyan
        (0, 0, 1),  # blue
        (1, 0, 1),  # magenta
    ]
)

# Display coordinate range (centres_x, centres_y are within 0-2000).
DISPLAY_SIZE = 2000.0

CHANNELS = "rgb"


def make_rgb_scales(gains=None, brightness=0):
    """
    (N_LEVELS, 3) float array of intensity values for each level and
    channel, at brightness setting `brightness`, with per-channel
    `gains` (three factors, default from DEFAULT_GAINS) applied after
    the low-level equalisation.
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


def to_uint8(scales):
    """Round and clip float intensities to uint8, warning if clipped."""
    rounded = np.rint(scales)
    if np.any(rounded > 255):
        print("Warning: some values exceed 255 and were clipped")
    return np.clip(rounded, 0, 255).astype("uint8")


def chart_cells(centres_x, centres_y):
    """
    (bar, level) grid cell of each LED: the chart is split into
    len(COL_BARS) equal-width columns and N_LEVELS equal-height rows.
    """
    nx = len(COL_BARS)
    bar = np.clip((centres_x / DISPLAY_SIZE * nx).astype(int), 0, nx - 1)
    level = np.clip(
        (centres_y / DISPLAY_SIZE * N_LEVELS).astype(int), 0, N_LEVELS - 1
    )
    return bar, level


def chart_rgb(bar, level, scales_u8):
    """(n_leds, 3) uint8 RGB values of the chart."""
    return (COL_BARS[bar] * scales_u8[level]).astype("uint8")


def print_scales(scales_u8, gains, brightness):
    print(
        f"\nBrightness {brightness}, gains "
        f"r={gains[0]:.3g} g={gains[1]:.3g} b={gains[2]:.3g}"
    )
    with np.printoptions(linewidth=200):
        for c, ch in enumerate(CHANNELS):
            print(f"  {ch}: {scales_u8[:, c]}")


def save_scales(scales_u8, gains, brightness):
    filename = f"colour_scales_b{brightness}.csv"
    header = (
        f"brightness={brightness}, gains r={gains[0]:.4g} "
        f"g={gains[1]:.4g} b={gains[2]:.4g}, params={SCALE_PARAMS}\n"
        "level,r,g,b"
    )
    data = np.column_stack([np.arange(N_LEVELS), scales_u8])
    np.savetxt(filename, data, fmt="%d", delimiter=",", header=header)
    print(f"Saved {filename}")


def parse_command(s, gains):
    """
    Apply one command string. Returns ("brightness", n), ("gains",
    new_gains), ("save", None), ("quit", None) or ("error", message).
    """
    parts = s.strip().lower().split()
    if not parts:
        return "error", "Empty command"
    if parts == ["q"]:
        return "quit", None
    if parts == ["s"]:
        return "save", None
    if len(parts) == 1 and parts[0].isdigit():
        n = int(parts[0])
        if 0 <= n < N_BRIGHTNESS:
            return "brightness", n
        return "error", f"Brightness must be 0-{N_BRIGHTNESS - 1}"
    if len(parts) == 2 and parts[0] in CHANNELS:
        try:
            gain = float(parts[1])
        except ValueError:
            return "error", f"Not a number: {parts[1]}"
        if gain < 0:
            return "error", "Gain must be >= 0"
        new_gains = list(gains)
        new_gains[CHANNELS.index(parts[0])] = gain
        return "gains", tuple(new_gains)
    return "error", f"Unknown command: {s.strip()}"


def main(brightness=DEFAULT_BRIGHTNESS, dry_run=False):
    from display1593.data.ledArray_data_1593 import centres_x, centres_y

    bar, level = chart_cells(np.asarray(centres_x), np.asarray(centres_y))
    gains = tuple(DEFAULT_GAINS[ch] for ch in CHANNELS)

    dis = None
    if not dry_run:
        from display1593 import Display1593

        dis = Display1593()
        dis.connect()

    try:
        while True:
            scales_u8 = to_uint8(make_rgb_scales(gains, brightness))
            print_scales(scales_u8, gains, brightness)
            if dis is not None:
                dis.set_all_leds(chart_rgb(bar, level, scales_u8))
                dis.show()

            while True:
                s = input(
                    "Brightness (0-7), 'r|g|b GAIN', 's' save, 'q' quit: "
                )
                action, value = parse_command(s, gains)
                if action == "error":
                    print(value)
                elif action == "save":
                    save_scales(scales_u8, gains, brightness)
                else:
                    break

            if action == "quit":
                break
            if action == "brightness":
                brightness = value
            elif action == "gains":
                gains = value
    finally:
        if dis is not None:
            dis.clear_all()
            dis.show()
            dis.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--brightness",
        type=int,
        default=DEFAULT_BRIGHTNESS,
        choices=range(N_BRIGHTNESS),
        help="initial brightness setting (default %(default)s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the scales without connecting to the display",
    )
    args = parser.parse_args()
    main(brightness=args.brightness, dry_run=args.dry_run)
