import threading
import time
import unittest

import numpy as np
import pytest

from display1593.data.ledArray_data_1593 import num_cells
from display1593.display1593 import (
    BoardSerialWorker,
    Display1593,
    SerialWorkerError,
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


def test_display1593_exposes_pipeline_api():
    display = Display1593()

    assert display.pipelined is False
    assert hasattr(display, "start_pipeline")
    assert hasattr(display, "stop_pipeline")
    assert hasattr(display, "flush")


class FakeBoard:
    """Fake serial connection to one board: replies to every command with
    the correct checksum (unless silent), optionally after a delay, and
    records each command received with its time."""

    def __init__(self, name, send_delay=0.0, silent=False):
        self.port = name
        self.send_delay = send_delay
        self.silent = silent
        self.received = []  # (time, command bytes)
        self.replies = []

    @property
    def in_waiting(self):
        return len(self.replies)

    def close(self):
        pass

    def commands(self):
        return [cmd for _, cmd in self.received]

    def time_of(self, cmd, occurrence=0):
        return [t for t, c in self.received if c == cmd][occurrence]


def _pipelined_display(monkeypatch, boards, **kwargs):
    def fake_send(ser, cmd):
        if ser.send_delay and bytes(cmd[:2]) != b"SN":
            time.sleep(ser.send_delay)
        ser.received.append((time.monotonic(), bytes(cmd)))
        if not ser.silent:
            ser.replies.append(calc_expected_response(cmd))

    def fake_receive(ser):
        return ser.replies.pop(0)

    monkeypatch.setattr(
        "display1593.display1593.send_data_to_arduino", fake_send
    )
    monkeypatch.setattr(
        "display1593.display1593.receive_data_from_arduino", fake_receive
    )
    display = Display1593(**kwargs)
    display._connections = boards
    display.start_pipeline()
    return display


def _colour(i):
    """A distinguishable 'set all LEDs to one colour' command, as sent."""
    return bytes((67, 65, i, 0, 0))


@pytest.fixture
def boards():
    return [FakeBoard("TEENSY1"), FakeBoard("TEENSY2")]


def test_pipelined_first_show_sends_frame_but_shows_nothing(
    monkeypatch, boards
):
    display = _pipelined_display(monkeypatch, boards)
    try:
        assert display.pipelined
        display.set_all_leds_one_colour((1, 0, 0))
        assert boards[0].commands() == []  # staged on the Pi until show()

        display.show()
        display._wait_for(
            lambda: all(w.is_idle() for w in display.serial_workers)
        )

        for board in boards:
            assert board.commands() == [_colour(1)]
    finally:
        display.stop_pipeline()


def test_pipelined_show_displays_previous_frame_at_t(monkeypatch, boards):
    display = _pipelined_display(monkeypatch, boards)
    try:
        display.set_all_leds_one_colour((1, 0, 0))
        display.show()
        display.set_all_leds_one_colour((2, 0, 0))
        t = time.monotonic() + 0.05
        display.show(t)
        assert time.monotonic() < t  # returned without waiting for t
        display.flush()

        for board in boards:
            # Frame 1 sent, shown at t, then frame 2 sent and (by
            # flush()) shown.
            assert board.commands() == [_colour(1), b"SN", _colour(2), b"SN"]
            sn_time = board.time_of(b"SN")
            assert t <= sn_time < t + 0.005
            assert board.time_of(_colour(2)) > sn_time
        sn_times = [board.time_of(b"SN") for board in boards]
        assert abs(sn_times[0] - sn_times[1]) < 0.005
    finally:
        display.stop_pipeline()


def test_pipelined_flush_shows_frame_set_after_last_show(
    monkeypatch, boards
):
    display = _pipelined_display(monkeypatch, boards)
    try:
        display.set_all_leds_one_colour((1, 0, 0))
        display.flush()

        for board in boards:
            assert board.commands() == [_colour(1), b"SN"]
    finally:
        display.stop_pipeline()


def test_pipelined_flush_shows_last_frame_at_t(monkeypatch, boards):
    display = _pipelined_display(monkeypatch, boards)
    try:
        display.set_all_leds_one_colour((1, 0, 0))
        display.show()
        t = time.monotonic() + 0.05
        display.flush(t)

        assert time.monotonic() >= t  # flush() waits until shown
        for board in boards:
            assert board.commands() == [_colour(1), b"SN"]
            assert t <= board.time_of(b"SN") < t + 0.005
    finally:
        display.stop_pipeline()


def test_pipelined_show_records_spare_time(monkeypatch, boards):
    display = _pipelined_display(monkeypatch, boards)
    try:
        display.set_all_leds_one_colour((1, 0, 0))
        display.show()
        display.show(time.monotonic() + 0.05)  # shows frame 1
        display.flush()

        # One entry (the show with a t); the frame was sent well before t.
        assert len(display.show_slack) == 1
        assert 0.03 < display.show_slack[0] <= 0.05
    finally:
        display.stop_pipeline()


def test_pipelined_warns_if_a_board_is_not_ready_by_show_time(
    monkeypatch, caplog
):
    boards = [FakeBoard("TEENSY1", send_delay=0.1), FakeBoard("TEENSY2")]
    display = _pipelined_display(monkeypatch, boards)
    try:
        display.set_all_leds_one_colour((1, 0, 0))
        display.show()
        t = time.monotonic() + 0.02
        display.show(t)
        display.flush()

        assert "Frame shown" in caplog.text
        assert "TEENSY1 not ready" in caplog.text
        # Both halves still shown together, once TEENSY1 was ready.
        sn_times = [board.time_of(b"SN") for board in boards]
        assert min(sn_times) > t + 0.05
        assert abs(sn_times[0] - sn_times[1]) < 0.005
    finally:
        display.stop_pipeline()


def test_pipelined_warns_if_show_called_after_t(
    monkeypatch, boards, caplog
):
    display = _pipelined_display(monkeypatch, boards)
    try:
        display.show(time.monotonic() - 0.5)
        assert "show() called" in caplog.text
    finally:
        display.stop_pipeline()


def test_pipelined_show_blocks_when_too_many_frames_ahead(
    monkeypatch, boards
):
    display = _pipelined_display(monkeypatch, boards, max_frames_ahead=2)
    try:
        t0 = time.monotonic()
        display.show(t0 + 0.1)  # first call: nothing to show yet
        display.show(t0 + 0.2)  # shows frame 1 at t0 + 0.2
        display.show(t0 + 0.3)  # shows frame 2 at t0 + 0.3
        assert time.monotonic() < t0 + 0.1  # none of those waited

        display.show(t0 + 0.4)  # 2 shows pending: waits for the first
        assert time.monotonic() >= t0 + 0.2
    finally:
        display.stop_pipeline()


def test_pipelined_show_raises_if_a_board_stops_replying(monkeypatch):
    boards = [FakeBoard("TEENSY1"), FakeBoard("TEENSY2", silent=True)]
    display = _pipelined_display(
        monkeypatch, boards, response_timeout=0.05
    )
    try:
        display.set_all_leds_one_colour((1, 0, 0))
        display.show()
        with pytest.raises(SerialWorkerError, match="TEENSY2") as exc_info:
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                display.show()
                time.sleep(0.01)
        assert isinstance(exc_info.value.__cause__, TimeoutError)
    finally:
        display.stop_pipeline()


def test_stop_pipeline_returns_to_synchronous_mode(monkeypatch, boards):
    display = _pipelined_display(monkeypatch, boards)
    display.stop_pipeline()

    assert display.pipelined is False
    assert display.serial_workers == []


def _sync_display(monkeypatch):
    """Display with two dummy connections and no workers (synchronous
    mode), recording the time each command is sent."""
    sent = []

    class DummySerial:
        in_waiting = 0

    def fake_send(ser, cmd):
        sent.append((time.monotonic(), bytes(cmd)))

    monkeypatch.setattr(
        "display1593.display1593.send_data_to_arduino", fake_send
    )
    monkeypatch.setattr(
        "display1593.display1593.check_response",
        lambda ser, cmd, timeout_after=1: None,
    )
    display = Display1593()
    display._connections = [DummySerial(), DummySerial()]
    return display, sent


def test_show_sync_sends_sn_to_both_boards_immediately(monkeypatch):
    display, sent = _sync_display(monkeypatch)

    display.show()

    assert [cmd for _, cmd in sent] == [b"SN", b"SN"]


def test_show_sync_waits_until_t(monkeypatch):
    display, sent = _sync_display(monkeypatch)

    t = time.monotonic() + 0.05
    display.show(t)

    assert [cmd for _, cmd in sent] == [b"SN", b"SN"]
    assert sent[0][0] >= t
    assert sent[0][0] - t < 0.005


def test_show_sync_records_spare_time(monkeypatch):
    display, sent = _sync_display(monkeypatch)

    display.show()  # no t: nothing recorded
    display.show(time.monotonic() + 0.03)
    display.show(time.monotonic() - 0.01)  # late

    assert len(display.show_slack) == 2
    assert 0.02 < display.show_slack[0] <= 0.03
    assert display.show_slack[1] < 0


def test_show_sync_warns_and_shows_immediately_if_t_passed(
    monkeypatch, caplog
):
    display, sent = _sync_display(monkeypatch)

    before = time.monotonic()
    display.show(before - 0.5)

    assert [cmd for _, cmd in sent] == [b"SN", b"SN"]
    assert sent[0][0] - before < 0.005
    assert "Show time missed" in caplog.text


if __name__ == "__main__":
    unittest.main()
