#!/usr/bin/env python3
"""Compare synchronous and pipelined display modes on the same animation.

Plays a dim scrolling rainbow (every LED updated every frame, via
set_all_leds()) in both modes and reports, for each mode:

1. Maximum frame rate, unpaced: frames are shown as fast as possible
   (show() with no t), with --compute-ms of simulated per-frame work
   (a busy-wait standing in for e.g. a simulation step). Synchronous
   mode can't overlap that work with sending, so expect roughly
   1 / (compute + send) frames per second, vs. 1 / max(compute, send)
   pipelined.
2. Paced playback at each --fps: show(t) on a fixed schedule, measuring
   how far each show command went out from its scheduled time t, the
   skew between the two boards, how long the script was blocked in the
   driver per frame, and how many late-frame warnings were logged.

Show times are measured on the Pi, when each SN command is handed to the
serial port - not when the LEDs actually change.

Run from the repo root on the Pi, with nothing else using the display
(stop the clock service first: sudo systemctl stop myscript.service):

    python diagnostics/compare_modes.py
    python diagnostics/compare_modes.py --fps 10,20,30,40 --compute-ms 30

--mock runs against fake boards (no hardware) to check the script itself;
its numbers say nothing about the real display.
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np

import display1593.display1593 as disp_mod
from display1593 import Display1593
from display1593.display1593 import SerialWorkerError
from display1593.logging_utils import configure_root_logging

LOG_PATH = Path("compare_modes.log")  # in the current directory

BRIGHTNESS = 16  # peak per channel, 0-255; kept low to protect the PSU
# Frames left out of the "blocked" stats: the first few show(t) calls
# wait out the 0.2 s start delay (pipelined mode: until max_frames_ahead
# is reached).
LEAD_IN = 4


class LogCounter(logging.Handler):
    """Count the driver's lateness warnings instead of printing them."""

    PATTERNS = {
        "frame_late": "Frame shown",  # pipelined: board not ready by t
        "call_late": "show() called",  # pipelined: show(t) after t
        "missed": "Show time missed",  # synchronous: show(t) after t
    }

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.other = []  # any other warnings, over the whole run
        self.reset()

    def reset(self):
        self.counts = dict.fromkeys(self.PATTERNS, 0)

    def emit(self, record):
        msg = record.getMessage()
        for key, pattern in self.PATTERNS.items():
            if msg.startswith(pattern):
                self.counts[key] += 1
                return
        self.other.append(msg)


class SNRecorder:
    """Wrap the driver's send function to timestamp every SN command."""

    def __init__(self):
        self.original = disp_mod.send_data_to_arduino
        self.times = {}  # port -> [time SN was handed to the port]
        disp_mod.send_data_to_arduino = self.send

    def send(self, ser, cmd):
        if cmd[0] == 83 and cmd[1] == 78:  # b"SN"
            self.times.setdefault(ser.port, []).append(time.monotonic())
        self.original(ser, cmd)

    def reset(self):
        self.times = {}


def install_mock(bytes_per_s=200_000):
    """Replace the serial layer with fake boards that reply correctly."""
    names = iter(list(disp_mod.NUMBER_OF_LEDS))

    class FakeSerial:
        def __init__(self, port, baudrate=None):
            self.port = port
            self.replies = []

        @property
        def in_waiting(self):
            return len(self.replies)

        def reset_input_buffer(self):
            pass

        def read(self, n=1):
            return b""

        def close(self):
            pass

    def fake_send(ser, cmd):
        time.sleep(0.0001 + len(cmd) / bytes_per_s)
        ser.replies.append(disp_mod.calc_expected_response(cmd))

    def fake_receive(ser):
        return ser.replies.pop(0)

    disp_mod.serial.Serial = FakeSerial
    disp_mod.connect_to_arduino = lambda ser: (0, next(names))
    disp_mod.send_data_to_arduino = fake_send
    disp_mod.receive_data_from_arduino = fake_receive


def busy_wait(seconds):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


def rainbow_frames(dis, n_frames):
    x = dis.leds.centres_x / dis.leds.centres_x.max()
    offsets = np.array([0, 1 / 3, 2 / 3])
    for k in range(n_frames):
        phase = x[:, None] + k / 50 - offsets
        rgb = BRIGHTNESS * (0.5 + 0.5 * np.cos(2 * np.pi * phase))
        yield rgb.astype(np.uint8)


def run_unpaced(dis, n_frames, compute_s):
    t0 = time.monotonic()
    for rgb in rainbow_frames(dis, n_frames):
        busy_wait(compute_s)
        dis.set_all_leds(rgb)
        dis.show()
    dis.flush()
    return n_frames / (time.monotonic() - t0)


def run_paced(dis, n_frames, fps, compute_s):
    """Returns (scheduled show times, blocked seconds per frame)."""
    period = 1 / fps
    t_show = time.monotonic() + 0.2
    scheduled = []
    blocked = []
    for rgb in rainbow_frames(dis, n_frames):
        busy_wait(compute_s)
        t0 = time.monotonic()
        dis.set_all_leds(rgb)
        dis.show(t_show)
        blocked.append(time.monotonic() - t0)
        scheduled.append(t_show)
        t_show += period
    dis.flush(t_show)
    if dis.pipelined:
        # Each show(t) displays the previous frame, the first call shows
        # nothing, and flush(t) shows the last frame.
        scheduled = scheduled[1:] + [t_show]
    return np.array(scheduled), np.array(blocked)


def run_paced_row(dis, mode, fps, n_frames, compute_s, recorder, counter):
    scheduled, blocked = run_paced(dis, n_frames, fps, compute_s)
    sn = [np.array(v) for v in recorder.times.values()]
    if len(sn) != 2 or any(len(s) != len(scheduled) for s in sn):
        print(
            f"  unexpected SN counts {[len(s) for s in sn]} for "
            f"{len(scheduled)} frames; skipping"
        )
        return None
    error_ms = 1000 * (np.concatenate(sn) - np.tile(scheduled, 2))
    skew_ms = 1000 * np.abs(sn[0] - sn[1])
    return (
        mode,
        fps,
        error_ms,
        skew_ms,
        blocked[LEAD_IN:] * 1000,
        dict(counter.counts),
    )


def stats(a):
    return f"{np.mean(a):6.1f} {np.percentile(a, 95):6.1f} {np.max(a):6.1f}"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--fps",
        default="10,20,30",
        help="comma-separated frame rates for the paced runs (default "
        "%(default)s)",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=5,
        help="duration of each paced run (default %(default)s)",
    )
    parser.add_argument(
        "--max-rate-frames",
        type=int,
        default=100,
        help="frames in each unpaced max-rate run (default %(default)s)",
    )
    parser.add_argument(
        "--compute-ms",
        type=float,
        default=20,
        help="simulated per-frame work in ms (default %(default)s)",
    )
    parser.add_argument(
        "--max-inflight",
        type=int,
        default=None,
        help="pipelined mode: commands each board may have sent but not "
        "yet acknowledged (default: the driver's, "
        f"{Display1593().max_inflight})",
    )
    parser.add_argument(
        "--modes",
        default="sync,pipelined",
        help="comma-separated modes to test (default %(default)s)",
    )
    parser.add_argument(
        "--mock", action="store_true", help="use fake boards, no hardware"
    )
    args = parser.parse_args()
    fps_list = [float(f) for f in args.fps.split(",")]
    compute_s = args.compute_ms / 1000

    configure_root_logging(LOG_PATH)
    logging.getLogger(__name__).info("Started: %s", vars(args))
    if args.mock:
        install_mock()
    recorder = SNRecorder()
    counter = LogCounter()
    logging.getLogger("display1593").addHandler(counter)

    all_modes = {"sync": False, "pipelined": True}
    modes = [(m, all_modes[m]) for m in args.modes.split(",")]
    max_rates = {}
    paced_rows = []
    failures = {}
    with Display1593() as dis:
        for mode, pipelined in modes:
            if pipelined:
                dis.start_pipeline(max_inflight=args.max_inflight)
            else:
                dis.stop_pipeline()
            try:
                print(f"{mode}: max rate ({args.max_rate_frames} frames)...")
                max_rates[mode] = run_unpaced(
                    dis, args.max_rate_frames, compute_s
                )
                for fps in fps_list:
                    n_frames = max(2, int(args.seconds * fps))
                    print(f"{mode}: {fps:g} fps ({n_frames} frames)...")
                    recorder.reset()
                    counter.reset()
                    row = run_paced_row(
                        dis, mode, fps, n_frames, compute_s, recorder, counter
                    )
                    if row is not None:
                        paced_rows.append(row)
            except SerialWorkerError as err:
                # The serial streams may now be out of step with the
                # boards, so don't try anything else in this mode.
                print(f"  FAILED: {err}")
                failures[mode] = err
        dis.stop_pipeline()
        try:
            dis.clear_all()
            dis.show()
        except Exception as err:
            print(f"Couldn't clear the display: {err}")

    print()
    print(
        f"Max frame rate, unpaced, with {args.compute_ms:g} ms simulated "
        "work per frame:"
    )
    for mode, _ in modes:
        rate = max_rates.get(mode)
        result = "failed" if rate is None else f"{rate:6.1f} frames/s"
        print(f"  {mode:>9}: {result}")

    print()
    print("Paced playback (ms; show time error = SN sent minus scheduled t):")
    print(
        f"{'mode':>9} {'fps':>4} | {'show time error':^20} | "
        f"{'board skew':^20} | {'blocked in driver':^20} | late warnings"
    )
    print(
        f"{'':>9} {'':>4} | {'mean    p95    max':^20} | "
        f"{'mean    p95    max':^20} | {'mean    p95    max':^20} |"
    )
    for mode, fps, error_ms, skew_ms, blocked_ms, counts in paced_rows:
        late = ", ".join(f"{k}={v}" for k, v in counts.items() if v) or "-"
        print(
            f"{mode:>9} {fps:4g} | {stats(error_ms):^20} | "
            f"{stats(skew_ms):^20} | {stats(blocked_ms):^20} | {late}"
        )
    print()
    print(
        f"blocked in driver: time per frame inside set_all_leds() + show()"
        f"\n(first {LEAD_IN} frames excluded). In sync mode this includes"
        " waiting for t;\nin pipelined mode it's only non-zero when the"
        " script is max_frames_ahead\nframes ahead."
    )
    for mode, err in failures.items():
        print(f"\n{mode} mode FAILED: {err}")
    print(f"\nFull log: {LOG_PATH.resolve()}")
    if counter.other:
        print(f"\nOther warnings logged ({len(counter.other)}), e.g.:")
        for msg in counter.other[:5]:
            print(f"  {msg}")


if __name__ == "__main__":
    main()
