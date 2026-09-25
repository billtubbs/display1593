#!/usr/bin/env python3
"""Probe the real serial response ordering on the Teensy boards.

This script is intentionally simple and deterministic:

1. Connect to the display.
2. Send a single command and read one response.
3. Send a pair of commands and read the first two responses.
4. Repeat with a larger burst, increasing the number of queued commands until
   the board starts to show a pattern (or until we hit a safe upper bound).

The goal is to answer one question unambiguously: do responses arrive in the
same order as commands, once debug packets are ignored?
"""

import time

import numpy as np

from display1593 import Display1593
from display1593.display1593 import calc_expected_response
from serial_comm import receive_data_from_arduino, send_data_to_arduino


def expect_reply(ser, cmd):
    send_data_to_arduino(ser, cmd)
    response = receive_data_from_arduino(ser)
    expected = calc_expected_response(cmd)
    print("CMD:", cmd)
    print("EXP:", expected)
    print("GOT:", response)
    print("MATCH:", bool(np.array_equal(response, expected)))
    print("---")
    return response


def main():
    with Display1593() as dis:
        print("Connected")
        ser = dis._connections[0]

        # 1 command
        cmd1 = np.array([0x01, 0x02, 0x03, 0x04], dtype=np.uint8)
        print("Single command check:")
        expect_reply(ser, cmd1)

        # 2 commands, read replies in order
        print("Two-command check:")
        cmd2a = np.array([0x10, 0x20, 0x30], dtype=np.uint8)
        cmd2b = np.array([0x40, 0x50, 0x60], dtype=np.uint8)
        send_data_to_arduino(ser, cmd2a)
        send_data_to_arduino(ser, cmd2b)

        r1 = receive_data_from_arduino(ser)
        r2 = receive_data_from_arduino(ser)
        print("EXP1:", calc_expected_response(cmd2a))
        print(
            "GOT1:",
            r1,
            "MATCH1:",
            bool(np.array_equal(r1, calc_expected_response(cmd2a))),
        )
        print("EXP2:", calc_expected_response(cmd2b))
        print(
            "GOT2:",
            r2,
            "MATCH2:",
            bool(np.array_equal(r2, calc_expected_response(cmd2b))),
        )
        print("---")

        # Burst test: keep sending commands until we get some mismatches or the
        # response stream stalls.
        print("Burst ordering check:")
        for n in [3, 5, 8, 12, 16]:
            commands = [
                np.array([n, i, (i * 7) % 256], dtype=np.uint8)
                for i in range(n)
            ]
            for cmd in commands:
                send_data_to_arduino(ser, cmd)
            responses = []
            for _ in commands:
                response = receive_data_from_arduino(ser)
                responses.append(response)
                print("n=", n, "reply:", response)
            print(
                "n=",
                n,
                "all matches:",
                all(
                    np.array_equal(resp, calc_expected_response(cmd))
                    for resp, cmd in zip(responses, commands)
                ),
            )
            print("---")

        print("Done")


if __name__ == "__main__":
    main()
