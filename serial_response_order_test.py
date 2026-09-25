#!/usr/bin/env python3
"""Probe whether the board replies in-order for valid command traffic.

This uses the same actual command bytes the firmware understands (for example
"SN", "LC", "L1", "LN"). Arbitrary arrays such as [1, 2, 3, 4] are invalid
commands and will legitimately trigger the Arduino's "Invalid command"
message; those do not tell us anything about ordering.
"""

import numpy as np

from display1593 import Display1593
from display1593.display1593 import calc_expected_response
from serial_comm import receive_data_from_arduino, send_data_to_arduino


VALID_COMMANDS = {
    "SN": np.array(list(b"SN"), dtype=np.uint8),
    "LC": np.array(list(b"LC"), dtype=np.uint8),
    "L1": np.array(list(b"L1"), dtype=np.uint8),
}


def expect_reply(ser, cmd):
    send_data_to_arduino(ser, cmd)
    response = receive_data_from_arduino(ser)
    expected = calc_expected_response(cmd)
    match = bool(np.array_equal(response, expected))
    print("CMD:", cmd)
    print("EXP:", expected)
    print("GOT:", response)
    print("MATCH:", match)
    print("---")
    assert match, f"expected {expected}, got {response} for command {cmd}"
    return response


def main():
    with Display1593() as dis:
        print("Connected")
        ser = dis._connections[0]

        print("Single command check:")
        expect_reply(ser, VALID_COMMANDS["SN"])

        print("Two-command check:")
        cmd2a = VALID_COMMANDS["SN"]
        cmd2b = VALID_COMMANDS["LC"]
        send_data_to_arduino(ser, cmd2a)
        send_data_to_arduino(ser, cmd2b)

        r1 = receive_data_from_arduino(ser)
        r2 = receive_data_from_arduino(ser)

        expected1 = calc_expected_response(cmd2a)
        expected2 = calc_expected_response(cmd2b)
        print("EXP1:", expected1)
        print("GOT1:", r1, "MATCH1:", bool(np.array_equal(r1, expected1)))
        print("EXP2:", expected2)
        print("GOT2:", r2, "MATCH2:", bool(np.array_equal(r2, expected2)))
        assert np.array_equal(r1, expected1)
        assert np.array_equal(r2, expected2)
        print("---")

        print("Burst ordering check:")
        burst_sizes = [3, 5, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192]
        for n in burst_sizes:
            commands = []
            for i in range(n):
                if i % 3 == 0:
                    cmd = VALID_COMMANDS["SN"]
                elif i % 3 == 1:
                    cmd = VALID_COMMANDS["LC"]
                else:
                    cmd = np.concatenate(
                        [
                            VALID_COMMANDS["L1"],
                            np.array([i % 251], dtype=np.uint8),
                        ]
                    )
                commands.append(cmd)
            for cmd in commands:
                send_data_to_arduino(ser, cmd)
            responses = []
            for _ in commands:
                response = receive_data_from_arduino(ser)
                responses.append(response)
                print("n=", n, "reply:", response)
            matches = [
                np.array_equal(resp, calc_expected_response(cmd))
                for resp, cmd in zip(responses, commands)
            ]
            print("n=", n, "all matches:", all(matches))
            assert all(matches), f"mismatch for n={n}: {matches}"
            print("---")

        print("All valid-command ordering checks passed.")


if __name__ == "__main__":
    main()
