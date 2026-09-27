#!/usr/bin/env python3
"""Report timing for the async serial worker pipeline.

This is a focused diagnostic for the current per-board worker design in
`display1593.display1593`. It measures the time spent in the main phases of a
large LED update burst:

- generating the commands in Python
- queueing them to serial workers
- the worker's send() call
- the worker's response wait/check
- the final show_now() frame commit

The script is intended for the Raspberry Pi with real hardware, but it also
supports a `--mock` mode so it can be run locally without a display attached.
"""

from __future__ import annotations

import argparse
import threading
import time
from statistics import mean

import numpy as np

import display1593.display1593 as disp_mod
from display1593 import Display1593


class DummySerial:
    """Minimal serial stub for local benchmarking without hardware."""

    def __init__(self, port_name):
        self.port = port_name
        self.in_waiting = 0
        self.closed = False
        self._buffer = bytearray()

    def write(self, data):
        self._buffer.extend(data)
        self.in_waiting = len(self._buffer)
        return len(data)

    def read(self, size=1):
        if not self._buffer:
            return b""
        out = bytes(self._buffer[:size])
        del self._buffer[:size]
        self.in_waiting = len(self._buffer)
        return out

    def close(self):
        self.closed = True


def _run_instrumented_worker_benchmark(
    display,
    *,
    n_leds,
    repeats,
    batch_size,
    mock_mode,
    debug_responses=False,
):
    """Run one benchmark sequence and return timing summaries."""
    send_times = []
    response_times = []
    queue_times = []
    generation_times = []
    show_now_times = []
    total_times = []
    lock = threading.Lock()

    original_send = disp_mod.send_data_to_arduino
    original_check = disp_mod.check_response
    original_receive = disp_mod.receive_data_from_arduino
    original_validate = disp_mod.BoardSerialWorker._validate_response

    debug_events = []
    seen_debug = 0
    seen_unknown = 0
    seen_mismatch = 0

    def timed_send(ser, cmd):
        t0 = time.perf_counter_ns()
        if mock_mode:
            time.sleep(0.0005)
        else:
            original_send(ser, cmd)
        elapsed = time.perf_counter_ns() - t0
        with lock:
            send_times.append(elapsed)

    def timed_check(ser, cmd, timeout_after=1):
        t0 = time.perf_counter_ns()
        if mock_mode:
            time.sleep(0.0006)
        else:
            original_check(ser, cmd, timeout_after=timeout_after)
        elapsed = time.perf_counter_ns() - t0
        with lock:
            response_times.append(elapsed)

    def logged_receive(ser):
        response = original_receive(ser)
        if debug_responses:
            print(
                f"RAW RESPONSE: {np.asarray(response, dtype=np.uint8).tolist()} "
                f"len={len(response)}"
            )
        return response

    def logged_validate(self, response, expected_response):
        nonlocal seen_debug, seen_unknown, seen_mismatch
        kind = disp_mod.classify_response(response)
        if kind == "debug":
            seen_debug += 1
            if debug_responses:
                print(
                    "DEBUG PACKET: "
                    f"expected={np.asarray(expected_response, dtype=np.uint8).tolist()} "
                    f"got={np.asarray(response, dtype=np.uint8).tolist()}"
                )
            return True
        if kind == "unknown":
            seen_unknown += 1
            if debug_responses:
                print(
                    "UNKNOWN PACKET: "
                    f"expected={np.asarray(expected_response, dtype=np.uint8).tolist()} "
                    f"got={np.asarray(response, dtype=np.uint8).tolist()}"
                )
            return False
        if kind == "checksum" and not np.array_equal(
            response, expected_response
        ):
            seen_mismatch += 1
            if debug_responses:
                print(
                    "MISMATCH: "
                    f"expected={np.asarray(expected_response, dtype=np.uint8).tolist()} "
                    f"got={np.asarray(response, dtype=np.uint8).tolist()}"
                )
            debug_events.append(
                {
                    "expected": np.asarray(
                        expected_response, dtype=np.uint8
                    ).copy(),
                    "got": np.asarray(response, dtype=np.uint8).copy(),
                }
            )
            return False
        return original_validate(self, response, expected_response)

    disp_mod.send_data_to_arduino = timed_send
    disp_mod.check_response = timed_check
    disp_mod.receive_data_from_arduino = logged_receive
    disp_mod.BoardSerialWorker._validate_response = logged_validate

    try:
        for rep in range(repeats):
            # Keep the benchmark safe on real hardware: the display is very
            # bright even at 32, so use a low brightness ceiling for the test
            # pattern to avoid tripping the power breakers.
            rgb = np.random.default_rng(42 + rep).integers(
                0, 8, size=(n_leds, 3), dtype=np.uint8
            )
            led_ids = np.arange(n_leds, dtype=np.int32)

            generation_start = time.perf_counter_ns()
            for led in led_ids:
                rgb_val = rgb[led]
                assert rgb_val.shape == (3,)
                _ = (led, rgb_val)
            generation_times.append(time.perf_counter_ns() - generation_start)

            queue_start = time.perf_counter_ns()
            for start in range(0, n_leds, batch_size):
                batch = led_ids[start : start + batch_size]
                rgb_batch = rgb[start : start + batch_size]
                display.set_leds(batch, rgb_batch)
            queue_times.append(time.perf_counter_ns() - queue_start)

            total_start = time.perf_counter_ns()
            show_start = time.perf_counter_ns()
            display.show_now()
            show_now_times.append(time.perf_counter_ns() - show_start)
            for worker in display.serial_workers:
                worker.queue.join()
            total_times.append(time.perf_counter_ns() - total_start)
    finally:
        disp_mod.send_data_to_arduino = original_send
        disp_mod.check_response = original_check
        disp_mod.receive_data_from_arduino = original_receive
        disp_mod.BoardSerialWorker._validate_response = original_validate

    return {
        "generation_ms": np.array(generation_times) / 1_000_000,
        "queue_ms": np.array(queue_times) / 1_000_000,
        "send_ms": np.array(send_times) / 1_000_000,
        "response_ms": np.array(response_times) / 1_000_000,
        "show_now_ms": np.array(show_now_times) / 1_000_000,
        "total_ms": np.array(total_times) / 1_000_000,
        "debug_events": debug_events,
        "seen_debug": seen_debug,
        "seen_unknown": seen_unknown,
        "seen_mismatch": seen_mismatch,
    }


def _run_mock_display(n_leds, repeats, batch_size, debug_responses=False):
    display = Display1593()
    display._connections = [
        DummySerial("dummy-ttyACM0"),
        DummySerial("dummy-ttyACM1"),
    ]
    display.start_serial_workers()
    try:
        return _run_instrumented_worker_benchmark(
            display,
            n_leds=n_leds,
            repeats=repeats,
            batch_size=batch_size,
            mock_mode=True,
            debug_responses=debug_responses,
        )
    finally:
        display.stop_serial_workers()
        for ser in display._connections:
            ser.close()


def run_benchmark(
    n_leds, repeats, batch_size, mock_mode, debug_responses=False
):
    """Run the benchmark and print a compact report."""
    display = None
    try:
        if mock_mode:
            display = Display1593()
            max_leds = display.n_leds
            n_leds = min(n_leds, max_leds)
            results = _run_mock_display(
                n_leds, repeats, batch_size, debug_responses=debug_responses
            )
        else:
            display = Display1593()
            max_leds = display.n_leds
            n_leds = min(n_leds, max_leds)
            display.connect()
            display.start_serial_workers()
            try:
                results = _run_instrumented_worker_benchmark(
                    display,
                    n_leds=n_leds,
                    repeats=repeats,
                    batch_size=batch_size,
                    mock_mode=False,
                    debug_responses=debug_responses,
                )
            finally:
                try:
                    display.clear_all()
                    display.show_now()
                finally:
                    display.stop_serial_workers()
                    display.disconnect()
    finally:
        if display is not None and not mock_mode:
            try:
                display.clear_all()
                display.show_now()
            except Exception:
                pass

    print("Serial worker timing benchmark")
    print("=" * 72)
    print(f"n_leds={n_leds}  repeats={repeats}  batch_size={batch_size}")
    print(f"mock_mode={mock_mode}  debug_responses={debug_responses}")
    print()

    if debug_responses:
        print(
            "response summary: "
            f"debug={results['seen_debug']} unknown={results['seen_unknown']} "
            f"mismatch={results['seen_mismatch']}"
        )
        if results["debug_events"]:
            for idx, evt in enumerate(results["debug_events"][:10]):
                print(
                    f"event {idx}: expected={evt['expected'].tolist()} got={evt['got'].tolist()}"
                )
        print()

    names = [
        ("generation", "generation_ms"),
        ("queueing", "queue_ms"),
        ("send", "send_ms"),
        ("response_check", "response_ms"),
        ("show_now", "show_now_ms"),
        ("total", "total_ms"),
    ]

    for label, key in names:
        values = results[key]
        if values.size == 0:
            print(f"{label:>16}: no data")
            continue
        print(
            f"{label:>16}: min={values.min():8.3f} ms   "
            f"mean={mean(values):8.3f} ms   "
            f"max={values.max():8.3f} ms"
        )

    total = results["total_ms"]
    if total.size:
        print()
        print(f"overall mean total time: {mean(total):8.3f} ms")


def main():
    parser = argparse.ArgumentParser(
        description="Measure the timing breakdown of the serial worker pipeline."
    )
    parser.add_argument(
        "--n-leds",
        type=int,
        default=2000,
        help="Number of LED updates to enqueue in each timing run.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Number of timing repeats to average over.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="How many LEDs to pack into each set_leds() batch.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run without hardware using dummy serial ports.",
    )
    parser.add_argument(
        "--debug-responses",
        action="store_true",
        help="Log every raw serial response, mismatch, and malformed packet.",
    )
    args = parser.parse_args()

    if args.n_leds <= 0:
        raise SystemExit("--n-leds must be > 0")
    if args.repeat <= 0:
        raise SystemExit("--repeat must be > 0")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be > 0")

    run_benchmark(
        n_leds=args.n_leds,
        repeats=args.repeat,
        batch_size=args.batch_size,
        mock_mode=args.mock,
        debug_responses=args.debug_responses,
    )


if __name__ == "__main__":
    main()
