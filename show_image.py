"""Display an image on the LED display."""

import argparse

from display1593 import Display1593


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="path to the image file")
    parser.add_argument(
        "dimness",
        nargs="?",
        type=float,
        default=8,
        help="brightness divisor - higher is dimmer (default: 8)",
    )
    args = parser.parse_args()

    with Display1593() as dis:
        dis.show_image(args.path, dimness=args.dimness)
        dis.show_now()


if __name__ == "__main__":
    main()
