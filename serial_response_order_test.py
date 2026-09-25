#!/usr/bin/env python3
"""Probe whether the board replies in-order for valid command traffic.

This uses the same actual command bytes the firmware understands (for example
"SN", "LC", "L1", "LN"). Arbitrary arrays such as [1, 2, 3, 4] are invalid
commands and will legitimately trigger the Arduino's "Invalid command"
message; those do not tell us anything about ordering.
"""

import time
from collections import deque

import numpy as np

from display1593 import Display1593
from display1593.display1593 import calc_expected_response
from serial_comm import receive_data_from_arduino, send_data_to_arduino


VALID_COMMANDS = {
    "SN": np.array(list(b"SN"), dtype=np.uint8),
    "LC": np.array(list(b"LC"), dtype=np.uint8),
    "L1": np.array(list(b"L1"), dtype=np.uint8),
}


def receive_with_timeout(ser, timeout=2.0):
    """Probe helper used only in this script.

    The core library intentionally keeps the blocking receive path because a
    timeout in the shared helper broke the board hello handshake. This wrapper
    adds a bounded wait only around the real receive call to diagnose burst
    behavior without changing the production API.
    """
    deadline = time.monotonic() + timeout
    while True:
        if ser.in_waiting > 0:
            return receive_data_from_arduino(ser)
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out waiting for reply after {timeout:.2f}s"
            )
        time.sleep(0.001)


def expect_reply(ser, cmd):
    send_data_to_arduino(ser, cmd)
    response = receive_data_from_arduino(ser)
    expected = calc_expected_response(cmd)
    match = bool(np.array_equal(response, expected))
    assert match, f"expected {expected}, got {response} for command {cmd}"
    return response


def make_command(i):
    if i % 3 == 0:
        return VALID_COMMANDS["SN"]
    if i % 3 == 1:
        return VALID_COMMANDS["LC"]
    led_index = (i * 7) % 256
    rgb = np.array(
        [(i * 17) % 256, (i * 29) % 256, (i * 43) % 256],
        dtype=np.uint8,
    )
    return np.concatenate(
        [
            np.array([ord("L"), ord("1")], dtype=np.uint8),
            np.array(
                [led_index // 256 % 256, led_index % 256], dtype=np.uint8
            ),
            rgb,
        ]
    )


def run_burst_in_order(ser, commands, max_inflight=4, timeout=2.0):
    """Send a bounded in-flight burst and read responses in FIFO order.

    This matches the production protocol we want to validate: keep at most
    ``max_inflight`` commands outstanding at once, and process replies as soon
    as they arrive instead of sending a full burst and only then draining the
    reply queue.
    """
    pending = deque(commands)
    inflight = deque()
    responses = []

    while pending or inflight:
        while pending and len(inflight) < max_inflight:
            cmd = pending.popleft()
            send_data_to_arduino(ser, cmd)
            inflight.append((cmd, calc_expected_response(cmd)))

        if not inflight:
            continue

        if ser.in_waiting > 0:
            response = receive_with_timeout(ser, timeout=timeout)
            cmd, expected = inflight.popleft()
            responses.append(response)
            if not np.array_equal(response, expected):
                raise AssertionError(
                    f"expected {expected}, got {response} for command {cmd}"
                )
        else:
            time.sleep(0.001)

    return responses


def ramped_send_probe(ser, max_commands=400, max_inflight=4, timeout=2.0):
    """Increase the send rate until the reply path becomes the bottleneck.

    The idea is to keep adding commands into the pipeline and measuring when
    the host-side send loop starts stalling on the board replies rather than on
    the sender itself. The command count at which this happens is the effective
    saturation point for this protocol.
    """
    results = []
    for burst_n in range(1, max_commands + 1):
        commands = [make_command(i) for i in range(burst_n)]
        t0 = time.perf_counter()
        try:
            responses = run_burst_in_order(
                ser, commands, max_inflight=max_inflight, timeout=timeout
            )
        except TimeoutError as exc:
            elapsed = time.perf_counter() - t0
            return (
                burst_n,
                elapsed,
                responses if "responses" in locals() else [],
            )

        elapsed = time.perf_counter() - t0
        results.append((burst_n, elapsed, len(responses)))

    return None, None, results


def main():
    with Display1593() as dis:
        ser = dis._connections[0]

        expect_reply(ser, VALID_COMMANDS["SN"])

        cmd2a = VALID_COMMANDS["SN"]
        cmd2b = VALID_COMMANDS["LC"]
        send_data_to_arduino(ser, cmd2a)
        send_data_to_arduino(ser, cmd2b)

        r1 = receive_data_from_arduino(ser)
        r2 = receive_data_from_arduino(ser)

        expected1 = calc_expected_response(cmd2a)
        expected2 = calc_expected_response(cmd2b)
        assert np.array_equal(r1, expected1)
        assert np.array_equal(r2, expected2)

        burst_n, elapsed, results = ramped_send_probe(
            ser, max_commands=200, max_inflight=4, timeout=2.0
        )
        if burst_n is None:
            print("probe completed without saturation up to max_commands=200")
            print(
                "last results:",
                results[-3:] if results else [],
            )
            return

        print(
            f"saturation observed at burst_n={burst_n}, elapsed={elapsed:.3f}s"
        )
        print("recent results:", results[-3:] if results else [])


if __name__ == "__main__":
    main()
