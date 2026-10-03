"""Play the precomputed fire animation on the display, in a loop.

Plays data/fire_frames.npz if present - the current 197.7s loop, made
with the gen-video-frames package from the led-fire-place repo's
parameters (~10 MB, not committed: copied to the Pi separately) -
otherwise the committed 30.5s clip in data/ (one CSV per frame). Frames
are played at FPS (slower than the original video's 30 fps, which looks
better).

Uses the display's pipelined mode by default; --sync uses synchronous
mode instead. Late frames are logged to play_fire_frames.log; a summary
is printed on Ctrl+C. Equivalent to the general-purpose
play_frames.py in the repo root, with the fireplace's data and log.
"""

import argparse
import logging
from pathlib import Path

from display1593 import Display1593
from display1593.logging_utils import configure_root_logging
from display1593.playback import load_frames, play_frames

DATA_DIR = Path(__file__).parent / "data"
NPZ_PATH = DATA_DIR / "fire_frames.npz"
LOG_PATH = Path(__file__).parent / "play_fire_frames.log"
FPS = 24

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=FPS,
        help="frame rate (default %(default)s; the original video is 30)",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="use synchronous mode instead of the pipeline",
    )
    args = parser.parse_args()

    configure_root_logging(LOG_PATH)
    logger.info("Started: %s", vars(args))
    frames = load_frames(NPZ_PATH if NPZ_PATH.exists() else DATA_DIR)
    with Display1593(pipelined=not args.sync) as dis:
        play_frames(dis, frames, args.fps, log_path=LOG_PATH)
