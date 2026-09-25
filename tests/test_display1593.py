import threading
import time
import unittest

import numpy as np

from display1593.data.ledArray_data_1593 import num_cells
from display1593.display1593 import BoardSerialWorker, Display1593


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

    class DummySerial:
        def __init__(self):
            self.closed = False

    def fake_send(ser, cmd):
        sent.append(cmd.copy())

    def fake_check(ser, cmd):
        processed.append(cmd.copy())

    monkeypatch.setattr(
        "display1593.display1593.send_data_to_arduino", fake_send
    )
    monkeypatch.setattr("display1593.display1593.check_response", fake_check)

    worker = BoardSerialWorker(DummySerial(), queue_size=8)
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
        assert all(np.array_equal(a, b) for a, b in zip(processed, sent))
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
