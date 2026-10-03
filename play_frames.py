"""Play a precomputed LED frame sequence on the display, in a loop.

FRAMES is an .npz file with array "led_frames", shape (n_frames, 1593, 3),
uint8 - e.g. made from a video with the gen-video-frames package - or a
directory of CSV files, one per frame. Frames are shown at --fps (which
needn't match the source video's frame rate: a lower rate plays it
slower).

Uses the display's pipelined mode by default; --sync uses synchronous
mode instead. Late frames are logged to play_frames.log; a summary is
printed on Ctrl+C.

Example:
    python play_frames.py data/phytoBlue_frames.npz --fps 24
"""

import argparse
import logging
from pathlib import Path

from display1593 import Display1593
from display1593.logging_utils import configure_root_logging
from display1593.playback import load_frames, play_frames

LOG_PATH = Path(__file__).parent / "play_frames.log"
FPS = 24

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("frames", help=".npz file or directory of CSVs")
    parser.add_argument(
        "--fps",
        type=float,
        default=FPS,
        help="frame rate (default %(default)s)",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="use synchronous mode instead of the pipeline",
    )
    args = parser.parse_args()

    configure_root_logging(LOG_PATH)
    logger.info("Started: %s", vars(args))
    frames = load_frames(args.frames)
    with Display1593(pipelined=not args.sync) as dis:
        play_frames(dis, frames, args.fps, log_path=LOG_PATH)
