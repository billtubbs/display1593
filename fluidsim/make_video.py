"""
Stitch the PNG frames written by view_fluidsim.py into an MP4 timelapse.

Frames are named `frame_<N>.png` with a zero-padded width chosen per run
(based on how many snapshots that run produces), so two runs with
different `--seconds`/`--save-interval` values dropped into the same
directory can both contain a "frame_0"-equivalent file at different
widths (e.g. `frame_00.png` and `frame_000.png`) - sorting by filename
text would misorder or silently collide these. This script instead
sorts by the numeric index parsed out of each filename and refuses to
proceed if it finds two files claiming the same index, rather than
silently interleaving frames from unrelated runs.

Requires the `ffmpeg` binary (e.g. `brew install ffmpeg`).

The `--fps` here is playback speed for the timelapse, unrelated to
view_fluidsim.py's/play_fluidsim.py's simulated/display timing - each
frame is one `--save-interval` of simulated time apart (a minute, by
view_fluidsim.py's default), so e.g. `--fps 10` plays 10 of those
per second.
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
DEFAULT_FRAMES_DIR = _HERE / "data"
_FRAME_RE = re.compile(r"(\d+)")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frames-dir",
        default=str(DEFAULT_FRAMES_DIR),
        help="directory containing frame_<N>.png files from one run of "
        "view_fluidsim.py",
    )
    parser.add_argument(
        "--pattern",
        default="frame_*.png",
        help="glob pattern (within --frames-dir) selecting the frames",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output video path (default: <frames-dir>/fluidsim.mp4)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="playback frame rate of the output video",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="x264 quality (lower = better/larger, 18 is visually "
        "near-lossless; 23 is x264's own default)",
    )
    return parser.parse_args()


def _numbered_frames(frames_dir, pattern):
    """
    (index, path) for every file matching `pattern`, sorted by the
    numeric index parsed from the filename. Raises if two files share
    an index (see module docstring) or if a match has no digits at all.
    """
    by_index = {}
    for path in frames_dir.glob(pattern):
        m = _FRAME_RE.search(path.stem)
        if not m:
            raise ValueError(
                f"'{path.name}' matched {pattern!r} but has no digits to "
                "sort by"
            )
        index = int(m.group(1))
        if index in by_index:
            raise ValueError(
                f"both '{by_index[index].name}' and '{path.name}' parse "
                f"to frame index {index} - looks like frames from more "
                "than one view_fluidsim.py run are mixed in "
                f"{frames_dir}; point --frames-dir at a single run's "
                "output (or clean out the stale files) rather than "
                "guessing which one is right"
            )
        by_index[index] = path
    return sorted(by_index.items())


def main():
    args = parse_args()

    if shutil.which("ffmpeg") is None:
        sys.exit(
            "ffmpeg not found on PATH - install it first (e.g. "
            "`brew install ffmpeg`)"
        )

    frames_dir = Path(args.frames_dir)
    frames = _numbered_frames(frames_dir, args.pattern)
    if not frames:
        sys.exit(f"no files matching {args.pattern!r} in {frames_dir}")

    out_path = (
        Path(args.out) if args.out else frames_dir / "fluidsim.mp4"
    )
    print(f"found {len(frames)} frames in {frames_dir}/")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        n_digits = len(str(len(frames) - 1))
        for i, (_, path) in enumerate(frames):
            (tmp_dir / f"frame_{i:0{n_digits}d}.png").symlink_to(
                path.resolve()
            )

        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(args.fps),
            "-i",
            str(tmp_dir / f"frame_%0{n_digits}d.png"),
            # Pad to even dimensions - libx264 requires them, and PNG
            # frame sizes here aren't guaranteed to already be even.
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-crf",
            str(args.crf),
            "-pix_fmt",
            "yuv420p",
            str(out_path),
        ]
        subprocess.run(cmd, check=True)

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
