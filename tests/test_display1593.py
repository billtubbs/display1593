import threading
import time
import unittest

import numpy as np

from display1593.data.ledArray_data_1593 import num_cells
from display1593.display1593 import (
    BoardSerialWorker,
    Display1593,
    calc_expected_response,
)


def test_response_classifier_distinguishes_debug_and_checksum():
    from display1593.display1593 import classify_response

    debug_resp = np.array(
        [
            0,
            0,
            73,
            110,
            118,
            97,
            108,
            105,
            100,
            32,
            99,
            111,
            109,
            109,
            97,
            110,
            100,
        ],
        dtype=np.uint8,
    )
    checksum_resp = np.array([0, 2, 0, 0, 0, 161], dtype=np.uint8)
    malformed = np.array([1, 2, 3], dtype=np.uint8)

    assert classify_response(debug_resp) == "debug"
    assert classify_response(checksum_resp) == "checksum"
    assert classify_response(malformed) == "unknown"


class NearestNeighboursAttributeTests(unittest.TestCase):
    def setUp(self):
        self.display = Display1593()

    def test_same_shape(self):
        self.assertEqual(
            self.display.nearest_neighbours.shape,
            self.display.nearest_neighbour_distances.shape,
        )

    def test_row_count_matches_expected_number_of_leds(self):
        self.assertEqual(self.display.nearest_neighbours.shape[0], num_cells)
        self.assertEqual(
            self.display.nearest_neighbour_distances.shape[0], num_cells
        )

    def test_nearest_neighbours_dtype_is_uint16(self):
        self.assertEqual(self.display.nearest_neighbours.dtype, np.uint16)


def test_board_serial_worker_preserves_queue_order(monkeypatch):
    sent = []
    processed = []
    expected_responses = []

    class DummySerial:
        def __init__(self):
            self.closed = False
            self.in_waiting = 0

    def fake_send(ser, cmd):
        sent.append(cmd.copy())
        expected_responses.append(calc_expected_response(cmd))
        ser.in_waiting = 1

    def fake_receive(ser):
        ser.in_waiting = 0
        return expected_responses.pop(0)

    def fake_validate(self, response, expected):
        processed.append(expected.copy())
        return True

    monkeypatch.setattr(
        "display1593.display1593.send_data_to_arduino", fake_send
    )
    monkeypatch.setattr(
        "display1593.display1593.receive_data_from_arduino", fake_receive
    )
    monkeypatch.setattr(
        "display1593.display1593.BoardSerialWorker._validate_response",
        fake_validate,
    )

    worker = BoardSerialWorker(
        DummySerial(), queue_size=8, max_inflight=1, response_timeout=0.5
    )
    worker.start()
    try:
        for cmd in [
            np.array([1, 2, 3], dtype=np.uint8),
            np.array([4, 5, 6], dtype=np.uint8),
            np.array([7, 8, 9], dtype=np.uint8),
        ]:
            worker.enqueue(cmd)

        deadline = time.monotonic() + 2
        while len(processed) < 3 and time.monotonic() < deadline:
            time.sleep(0.01)

        assert len(sent) == 3
        assert all(
            np.array_equal(cmd, expected)
            for cmd, expected in zip(
                sent,
                [
                    np.array([1, 2, 3], dtype=np.uint8),
                    np.array([4, 5, 6], dtype=np.uint8),
                    np.array([7, 8, 9], dtype=np.uint8),
                ],
            )
        )
        expected_processed = [calc_expected_response(cmd) for cmd in sent]
        assert all(
            np.array_equal(a, b) for a, b in zip(processed, expected_processed)
        )
    finally:
        worker.shutdown()
        worker.join(timeout=1)


def test_board_serial_worker_rejects_corrupt_response_after_two_commands(
    monkeypatch,
):
    """Two queued commands are the smallest reproducer for a bad payload.

    The board is single-threaded and enforces FIFO reply order, so the real host
    bug is not out-of-order replies. The minimal valid regression is still two
    commands in flight, followed by a malformed or corrupted response payload.
    The host must reject that bad response instead of accepting it as a valid
    checksum for the oldest outstanding command.
    """
    seen = []

    class DummySerial:
        def __init__(self):
            self.in_waiting = 0
            self.pending = []

    first = np.array([1, 2, 3], dtype=np.uint8)
    second = np.array([4, 5, 6], dtype=np.uint8)
    expected_first = calc_expected_response(first)
    expected_second = calc_expected_response(second)
    corrupt = np.array([0, 3, 0, 0, 0, 15], dtype=np.uint8)

    def fake_send(ser, cmd):
        ser.pending.append(cmd.copy())
        ser.in_waiting += 1

    def fake_receive(ser):
        ser.in_waiting = max(0, ser.in_waiting - 1)
        return responses.pop(0)

    def fake_validate(self, response, expected):
        seen.append((response.copy(), expected.copy()))
        return np.array_equal(response, expected)

    monkeypatch.setattr(
        "display1593.display1593.send_data_to_arduino", fake_send
    )
    monkeypatch.setattr(
        "display1593.display1593.receive_data_from_arduino", fake_receive
    )
    monkeypatch.setattr(
        "display1593.display1593.BoardSerialWorker._validate_response",
        fake_validate,
    )

    serial = DummySerial()
    responses = [corrupt, expected_second]
    worker = BoardSerialWorker(serial, queue_size=8, max_inflight=2)
    worker.start()

    try:
        worker.enqueue(first)
        worker.enqueue(second)

        deadline = time.monotonic() + 1
        while len(seen) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)

        assert len(seen) == 2
        assert not np.array_equal(seen[0][0], seen[0][1])
        assert np.array_equal(seen[1][0], seen[1][1])
    finally:
        worker.shutdown()
        worker.join(timeout=1)


def test_receive_data_from_arduino_can_decode_impossible_644_byte_payload():
    """Minimal stale-frame reproducer for impossible length values.

    This does not prove the board sent a 644-byte reply. It demonstrates the
    exact host-side failure mode: if the serial stream starts mid-frame, or a
    stale packet is left in the buffer, then a short-command exchange can still
    decode to a 644-byte payload. That is how impossible response lengths like
    644 can appear without any out-of-order reply being possible.
    """

    class FakeSerial:
        def __init__(self):
            self._buf = b"\xfe" + b"\x00" * 644 + b"\xff"

        def read_until(self, marker, size=None):
            if marker == b"\xfe":
                out = b"\xfe"
                self._buf = self._buf[1:]
                return out
            if marker == b"\xff":
                out = self._buf
                self._buf = b""
                return out
            raise AssertionError(f"unexpected marker: {marker!r}")

    data = __import__("serial_comm").receive_data_from_arduino(FakeSerial())
    assert len(data) == 644
    assert data.shape == (644,)


def test_board_serial_worker_timeout_when_reply_stalls(monkeypatch):
    class DummySerial:
        def __init__(self):
            self.in_waiting = 0

    def fake_send(ser, cmd):
        pass

    monkeypatch.setattr(
        "display1593.display1593.send_data_to_arduino", fake_send
    )

    worker = BoardSerialWorker(
        DummySerial(), queue_size=8, max_inflight=1, response_timeout=0.02
    )
    worker.start()
    worker.enqueue(np.array([9, 9, 9], dtype=np.uint8))

    deadline = time.monotonic() + 1
    while worker.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)

    worker.join(timeout=1)
    assert not worker.is_alive()
    assert isinstance(worker.last_error, TimeoutError)


def test_board_serial_worker_uses_bounded_inflight_fifo(monkeypatch):
    sent = []
    responses = []

    class DummySerial:
        def __init__(self):
            self.in_waiting = 0
            self._pending = [
                np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8),
                np.array([2, 0, 0, 0, 0, 0], dtype=np.uint8),
                np.array([3, 0, 0, 0, 0, 0], dtype=np.uint8),
            ]

    class DummySerialLegacy:
        def __init__(self):
            self._pending = [
                np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8),
                np.array([2, 0, 0, 0, 0, 0], dtype=np.uint8),
                np.array([3, 0, 0, 0, 0, 0], dtype=np.uint8),
            ]

    def fake_send(ser, cmd):
        sent.append(cmd.copy())
        ser.in_waiting = 1

    def fake_receive(ser):
        ser.in_waiting = 0
        return responses.pop(0)

    def fake_validate(self, response, expected):
        return True

    monkeypatch.setattr(
        "display1593.display1593.send_data_to_arduino", fake_send
    )
    monkeypatch.setattr(
        "display1593.display1593.receive_data_from_arduino", fake_receive
    )
    monkeypatch.setattr(
        "display1593.display1593.BoardSerialWorker._validate_response",
        fake_validate,
    )

    serial = DummySerial()
    responses = serial._pending
    worker = BoardSerialWorker(serial, queue_size=8, max_inflight=3)
    worker.start()

    try:
        for cmd in [
            np.array([1, 2, 3], dtype=np.uint8),
            np.array([4, 5, 6], dtype=np.uint8),
            np.array([7, 8, 9], dtype=np.uint8),
        ]:
            worker.enqueue(cmd)

        deadline = time.monotonic() + 0.5
        while len(sent) < 3 and time.monotonic() < deadline:
            time.sleep(0.01)

        assert len(sent) == 3
        assert all(
            np.array_equal(a, b)
            for a, b in zip(
                sent,
                [
                    np.array([1, 2, 3], dtype=np.uint8),
                    np.array([4, 5, 6], dtype=np.uint8),
                    np.array([7, 8, 9], dtype=np.uint8),
                ],
            )
        )
    finally:
        worker.shutdown()
        worker.join(timeout=1)


def test_display1593_exposes_async_serial_setup():
    display = Display1593()

    assert hasattr(display, "serial_workers")
    assert hasattr(display, "start_serial_workers")
    assert hasattr(display, "submit_serial_command")


def test_show_now_uses_active_serial_workers(monkeypatch):
    class DummySerial:
        def __init__(self):
            self.in_waiting = 0

    sent = []
    processed = []

    def fake_send(ser, cmd):
        sent.append(cmd.copy())

    def fake_check(ser, cmd):
        processed.append(cmd.copy())

    monkeypatch.setattr(
        "display1593.display1593.send_data_to_arduino", fake_send
    )
    monkeypatch.setattr("display1593.display1593.check_response", fake_check)

    display = Display1593()
    display._connections = [DummySerial(), DummySerial()]
    display.serial_workers = [
        BoardSerialWorker(display._connections[0]),
        BoardSerialWorker(display._connections[1]),
    ]
    for worker in display.serial_workers:
        worker.start()

    try:
        display.show_now()
        deadline = time.monotonic() + 2
        while len(processed) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)

        assert len(sent) == 2
        assert all(
            np.array_equal(cmd, np.array(list(b"SN"), dtype=np.uint8))
            for cmd in sent
        )
    finally:
        for worker in display.serial_workers:
            worker.shutdown()
            worker.join(timeout=1)


def test_submit_frame_stages_commands_for_each_board(monkeypatch):
    sent = []

    class DummySerial:
        def __init__(self):
            self.in_waiting = 0

    def fake_send(ser, cmd):
        sent.append(cmd.copy())

    def fake_check(ser, cmd):
        return None

    monkeypatch.setattr(
        "display1593.display1593.send_data_to_arduino", fake_send
    )
    monkeypatch.setattr("display1593.display1593.check_response", fake_check)

    display = Display1593()
    display._connections = [DummySerial(), DummySerial()]
    display.serial_workers = [
        BoardSerialWorker(display._connections[0]),
        BoardSerialWorker(display._connections[1]),
    ]
    for worker in display.serial_workers:
        worker.start()

    try:
        display.submit_frame(
            {
                0: [np.array([1, 2, 3], dtype=np.uint8)],
                1: [np.array([4, 5, 6], dtype=np.uint8)],
            }
        )
        time.sleep(0.1)
        assert len(sent) == 2
    finally:
        for worker in display.serial_workers:
            worker.shutdown()
            worker.join(timeout=1)


if __name__ == "__main__":
    unittest.main()
