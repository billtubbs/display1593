"""Play the precomputed fire animation on the display, in a loop.

The animation is a sequence of per-LED RGB frames (one CSV per frame in
data/), made to be played at --source-fps. The display can run at a
higher --fps: in-between frames are then blended linearly from the two
nearest source frames, so the fire moves at the same speed but more
smoothly.

Uses the display's pipelined mode by default (each show(t) displays the
previous frame at t while the next one is sent), which can sustain a
much higher frame rate than synchronous mode (--sync). Late frames are
logged to play_fire_frames.log; a summary is printed on Ctrl+C.
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from display1593 import Display1593
from display1593.logging_utils import configure_root_logging

DATA_DIR = Path(__file__).parent / "data"
LOG_PATH = Path(__file__).parent / "play_fire_frames.log"
SOURCE_FPS = 16  # rate the frames in DATA_DIR were made for

logger = logging.getLogger(__name__)


class WarningCounter(logging.Handler):
    """Counts the display driver's warnings (e.g. late frames)."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.count = 0

    def emit(self, record):
        self.count += 1


def load_led_frames(data_dir):
    """Loads precomputed LED RGB frames (one CSV per frame) from data_dir."""
    frame_paths = sorted(Path(data_dir).glob("*.csv"))
    return np.stack(
        [np.loadtxt(p, delimiter=",", dtype="uint8") for p in frame_paths]
    )


def blended_frame(frames, position):
    """Frame at a fractional position in the (looping) animation, blended
    linearly between the two nearest source frames."""
    i = int(position) % len(frames)
    frac = position - int(position)
    if frac == 0:
        return frames[i]
    j = (i + 1) % len(frames)
    z = (1 - frac) * frames[i] + frac * frames[j]
    return np.rint(z).astype("uint8")


def main(dis, frames, fps, source_fps):
    period = 1 / fps
    step = source_fps / fps  # source frames per displayed frame
    print(
        f"Playing {len(frames)} frames at {fps:g} fps "
        f"({'pipelined' if dis.pipelined else 'synchronous'} mode, "
        f"source {source_fps:g} fps). Ctrl+C to stop."
    )

    counter = WarningCounter()
    logging.getLogger("display1593").addHandler(counter)

    dis.clear_all()
    position = 0.0
    n_shown = 0
    start_time = time.monotonic()
    t_show = start_time + 0.1
    try:
        while True:
            dis.set_all_leds(blended_frame(frames, position))
            # Pipelined: shows the previous frame at t_show and starts
            # sending this one. Synchronous: waits until t_show, then
            # shows this one.
            dis.show(t_show)
            n_shown += 1
            position += step
            t_show += period
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        logging.getLogger("display1593").removeHandler(counter)

    elapsed = time.monotonic() - start_time
    summary = (
        f"{n_shown} frames in {elapsed:.1f} s "
        f"({n_shown / elapsed:.1f} fps, target {fps:g}); "
        f"{counter.count} late-frame warnings (see {LOG_PATH.name})"
    )
    print(summary)
    logger.info(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=SOURCE_FPS,
        help="display frame rate (default %(default)s)",
    )
    parser.add_argument(
        "--source-fps",
        type=float,
        default=SOURCE_FPS,
        help="rate the animation frames were made for (default %(default)s)",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="use synchronous mode instead of the pipeline",
    )
    args = parser.parse_args()

    configure_root_logging(LOG_PATH)
    logger.info("Started: %s", vars(args))
    frames = load_led_frames(DATA_DIR)
    with Display1593(pipelined=not args.sync) as dis:
        main(dis, frames, args.fps, args.source_fps)
