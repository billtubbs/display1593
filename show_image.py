"""Display an image on the LED display."""

import argparse

from display1593 import Display1593


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="path to the image file")
    parser.add_argument(
        "--brightness",
        type=float,
        default=0,
        help="colour correction brightness setting (0-7 is the "
        "calibrated range, x1.15 per step; below 0 is dimmer; "
        "default: %(default)s)",
    )
    parser.add_argument(
        "--dimness",
        type=float,
        default=None,
        help="use the older uncorrected mapping instead (mean pixel "
        "value squared / (256 * DIMNESS); 8 was the old default)",
    )
    args = parser.parse_args()

    with Display1593() as dis:
        if args.dimness is not None:
            dis.show_image(args.path, dimness=args.dimness)
        else:
            dis.show_image(args.path, brightness=args.brightness)
        dis.show()


if __name__ == "__main__":
    main()
