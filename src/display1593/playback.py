"""
Playing precomputed LED frame sequences on the display, in a loop.

Frames are (n_frames, 1593, 3) uint8 RGB values, as sent to
set_all_leds() - e.g. generated from a video with the gen-video-frames
package (an .npz file with array "led_frames"), or the older one CSV per
frame format (fireplace/data/).

Each frame is scheduled with show(t) at a fixed rate, so frame times stay
accurate in either mode; pipelined mode (the default for the entry-point
scripts) has plenty of headroom - see README.md.
"""

import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Delay before the first frame is shown, from the start of play_frames()
START_DELAY = 0.1


class WarningCounter(logging.Handler):
    """Counts the display driver's warnings (e.g. late frames)."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.count = 0

    def emit(self, record):
        self.count += 1


def load_frames(path):
    """
    Loads LED frames from an .npz file (array "led_frames") or from a
    directory of CSV files (one per frame, 1593 rows of R,G,B, in filename
    order). Returns a uint8 array, shape (n_frames, 1593, 3).
    """
    path = Path(path)
    if path.is_dir():
        frame_paths = sorted(path.glob("*.csv"))
        logger.info("Loading %d CSV frames from %s", len(frame_paths), path)
        frames = np.array(
            [np.loadtxt(p, delimiter=",", dtype="uint8") for p in frame_paths]
        )
    else:
        logger.info("Loading frames from %s", path)
        frames = np.load(path)["led_frames"]
    if frames.ndim != 3 or frames.shape[1:] != (1593, 3) or not len(frames):
        raise ValueError(
            f"Expected frames of shape (n_frames, 1593, 3) from '{path}', "
            f"got {frames.shape}"
        )
    return frames.astype("uint8", copy=False)


def play_frames(dis, frames, fps, max_frames=None, log_path=None):
    """
    Plays frames on a connected display at fps, looping, until Ctrl+C (or
    max_frames frames have been shown). Then prints and logs a summary:
    the frame rate achieved, late-frame warnings from the driver, and the
    spare time per frame (dis.show_slack).

    :param dis: connected Display1593 (pipelined or synchronous mode).
    :param frames: array or list of (1593, 3) uint8 frames.
    :param fps: frame rate.
    :param max_frames: stop after this many frames (default: never).
    :param log_path: log file to point to in the summary, if any.
    :return: the summary text.
    """
    period = 1 / fps
    print(
        f"Playing {len(frames)} frames at {fps:g} fps "
        f"({'pipelined' if dis.pipelined else 'synchronous'} mode). "
        "Ctrl+C to stop."
    )

    counter = WarningCounter()
    logging.getLogger("display1593").addHandler(counter)

    dis.clear_all()
    dis.show_slack.clear()
    n_shown = 0
    start_time = time.monotonic()
    t_show = start_time + START_DELAY
    try:
        while max_frames is None or n_shown < max_frames:
            dis.set_all_leds(frames[n_shown % len(frames)])
            # Pipelined: shows the previous frame at t_show and starts
            # sending this one. Synchronous: waits until t_show, then
            # shows this one.
            dis.show(t_show)
            n_shown += 1
            t_show += period
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        logging.getLogger("display1593").removeHandler(counter)

    elapsed = time.monotonic() - start_time
    summary = (
        f"{n_shown} frames in {elapsed:.1f} s "
        f"({n_shown / elapsed:.1f} fps, target {fps:g}); "
        f"{counter.count} late-frame warnings"
    )
    if log_path is not None:
        summary += f" (see {Path(log_path).name})"
    # Spare time per frame (negative = late): pipelined mode, between
    # both boards having received the frame and its show time; synchronous
    # mode, between the show(t) call and t. The first frame is left out:
    # its spare time includes the start-up delay.
    if len(dis.show_slack) > 1:
        slack_ms = 1000 * np.array(dis.show_slack)[1:]
        summary += (
            f"\nSpare time per frame over the last {len(slack_ms)} "
            f"frames: mean {slack_ms.mean():.1f} ms, "
            f"shortest {slack_ms.min():.1f} ms "
            f"(of {1000 * period:.1f} ms per frame)"
        )
    print(summary)
    logger.info(summary)
    return summary
