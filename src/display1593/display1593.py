import logging
import threading
import time
from collections import deque
from itertools import pairwise
from pathlib import Path
from queue import Empty, Queue

import numpy as np
import serial
from numba import jit, types
from PIL import Image
from serial_comm import (
    connect_to_arduino,
    receive_data_from_arduino,
    send_data_to_arduino,
)

from display1593.data.ledArray_data_1593 import centres_x, centres_y
from display1593.image_conversion import convert_image as _convert_image
from display1593.image_conversion import prepare_image as _prepare_image
from display1593.lock import DEFAULT_TIMEOUT as DEFAULT_LOCK_TIMEOUT
from display1593.lock import DisplayLock, DisplayLockTimeout

# The nearest_neighbours/nearest_neighbour_distances arrays in
# ledArray_data_1593.py contain indexing errors for LEDs near the edges
# of the array (see the comment above those arrays for details), so the
# corrected data is loaded here from CSV files instead. These were
# generated with compute_nearest_neighbours() in ledArray_data_1593.py.
_DATA_DIR = Path(__file__).parent / "data"
nearest_neighbours = np.loadtxt(
    _DATA_DIR / "nearest_neighbours_1593.csv", delimiter=",", dtype=np.uint16
)
nearest_neighbour_distances = np.loadtxt(
    _DATA_DIR / "nearest_neighbour_distances_1593.csv", delimiter=","
)

# Numba array types
readonly_uint8_array = types.Array(types.uint8, 1, "C", readonly=True)
writable_uint8_array = types.Array(types.uint8, 1, "C")
int32_array = types.Array(types.int32, 1, "C")
int32_array_1d = types.Array(types.int32, 1, "C")
uint8_array_2d = types.Array(types.uint8, 2, "C")


# Log records are handled by whichever entry-point script imports this
# module - see display1593.logging_utils.configure_root_logging().
logger = logging.getLogger(__name__)


def _drain_serial_input(ser, timeout=0.25):
    """Discard stale bytes in the local serial RX buffer.

    This is a Python-side mitigation for residual bytes left behind from an
    earlier session or a previous failed burst. It cannot clear bytes the
    Arduino is still actively sending, but it does remove any stale traffic that
    is already queued in the local USB driver before the board hello is read.
    """
    try:
        if hasattr(ser, "reset_input_buffer"):
            ser.reset_input_buffer()
    except (AttributeError, OSError, serial.SerialException):
        pass

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if getattr(ser, "in_waiting", 0) <= 0:
                break
            ser.read(getattr(ser, "in_waiting", 0))
        except (AttributeError, OSError, serial.SerialException):
            break


COMMAND_LC = np.array(list(b"LC"), dtype=np.uint8)  # implemented
COMMAND_SN = np.array(list(b"SN"), dtype=np.uint8)

B2 = 32
BLACK = np.zeros(3, dtype="uint8")
WHITE = np.full_like(BLACK, B2)
RED = np.array([B2, 0, 0], dtype="uint8")
GREEN = np.array([0, B2, 0], dtype="uint8")
BLUE = np.array([0, 0, B2], dtype="uint8")
YELLOW = np.array([B2, B2, 0], dtype="uint8")
MAGENTA = np.array([B2, 0, B2], dtype="uint8")
CYAN = np.array([0, B2, B2], dtype="uint8")

# Arduino communication
BAUD_RATE = 57600


# Serial ports of Teensy devices
# Find these by running ls /dev/tty.* from command line
# SERIAL_PORTS = {
#     49: '/dev/cu.usbmodem12745401',
#     50: '/dev/cu.usbmodem6862001'
# }
# Usually,
#  - TEENSY1 is on usb port 1275401
#  - TEENSY2 is on usb port 6862001
# Raspberry Pi uses the /dev/ttyACM* naming scheme
# Find these by running ls /dev/tty.* from command line
SERIAL_PORTS = ["/dev/ttyACM0", "/dev/ttyACM1"]
# Usually (but not always),
#  - TEENSY1 is on usb port '/dev/ttyACM1'
#  - TEENSY2 is on usb port '/dev/ttyACM0'

# LED setup
NUMBER_OF_LEDS = {"TEENSY1": 798, "TEENSY2": 795}


@jit(
    [types.uint8[:, :](types.int32[:]), types.uint8[:, :](types.int64[:])],
    nopython=True,
    cache=True,
)
def make_idx_array(leds):
    idx = np.empty((leds.shape[0], 2), dtype=np.uint8)
    for i in range(leds.shape[0]):
        idx[i, 0] = leds[i] // 256 % 256
        idx[i, 1] = leds[i] % 256
    # idx = np.array([(i // 256 % 256, i % 256) for i in leds], dtype=np.uint8)
    return idx


@jit(
    [types.Tuple((int32_array, int32_array))(int32_array, int32_array)],
    nopython=True,
    cache=True,
)
def _board_leds(leds, led_idx):
    """Filter led ids into separate lists for each board."""
    n = len(leds)

    # Pre-allocate with maximum possible size
    board_leds_0 = np.empty(n, dtype=leds.dtype)
    board_leds_1 = np.empty(n, dtype=leds.dtype)

    # Fill arrays and track actual sizes
    idx0 = 0
    idx1 = 0
    for i in range(n):
        led = leds[i]
        if led < led_idx[1]:
            board_leds_0[idx0] = led - led_idx[0]
            idx0 += 1
        elif led < led_idx[2]:
            board_leds_1[idx1] = led - led_idx[1]
            idx1 += 1
        else:
            raise ValueError("invalid led id")

    # Trim to actual sizes
    board_leds_0 = board_leds_0[:idx0]
    board_leds_1 = board_leds_1[:idx1]

    return board_leds_0, board_leds_1


return_type = types.Tuple(
    (int32_array_1d, int32_array_1d, uint8_array_2d, uint8_array_2d)
)


@jit(
    [return_type(int32_array_1d, uint8_array_2d, int32_array_1d)],
    nopython=True,
    cache=True,
)
def _board_leds_with_rgb(leds, rgb_array, led_idx):
    """Filter led ids into separate lists for each board."""
    n = len(leds)

    # Pre-allocate with maximum possible size
    board_leds_0 = np.empty(n, dtype=leds.dtype)
    board_leds_1 = np.empty(n, dtype=leds.dtype)
    rgb_arrays_0 = np.empty((n, 3), dtype=np.uint8)
    rgb_arrays_1 = np.empty((n, 3), dtype=np.uint8)

    # Fill arrays and track actual sizes
    idx0 = 0
    idx1 = 0
    for i in range(n):
        led = leds[i]
        if led < led_idx[1]:
            board_leds_0[idx0] = led - led_idx[0]
            rgb_arrays_0[idx0] = rgb_array[i]
            idx0 += 1
        elif led < led_idx[2]:
            board_leds_1[idx1] = led - led_idx[1]
            rgb_arrays_1[idx1] = rgb_array[i]
            idx1 += 1
        else:
            raise ValueError("invalid led id")

    # Trim to actual sizes
    board_leds_0 = board_leds_0[:idx0]
    board_leds_1 = board_leds_1[:idx1]
    rgb_arrays_0 = rgb_arrays_0[:idx0]
    rgb_arrays_1 = rgb_arrays_1[:idx1]

    return board_leds_0, board_leds_1, rgb_arrays_0, rgb_arrays_1


@jit(
    [
        writable_uint8_array(readonly_uint8_array),
        writable_uint8_array(writable_uint8_array),
    ],
    nopython=True,
    cache=True,
)
def calc_expected_response(cmd):
    """
    Calculate the expected response of the Arduino to the command.

    Args:
        cmd: NumPy array of uint8 values

    Returns:
        NumPy array of 6 uint8 values:
        - Bytes 0-1: length of cmd (16-bit big-endian)
        - Bytes 2-5: sum of cmd values (32-bit big-endian)
    """
    expected_response = np.empty(6, dtype=np.uint8)

    # Get the length of cmd
    cmd_length = len(cmd)

    # Bytes 0-1: length as 16-bit big-endian (high byte first)
    expected_response[0] = (cmd_length >> 8) & 0xFF  # High byte
    expected_response[1] = cmd_length & 0xFF  # Low byte

    # Calculate sum of all values in cmd
    cmd_sum = np.uint32(0)
    for i in range(len(cmd)):
        cmd_sum += cmd[i]

    # Bytes 2-5: sum as 32-bit big-endian (high byte first)
    expected_response[2] = (cmd_sum >> 24) & 0xFF  # Highest byte
    expected_response[3] = (cmd_sum >> 16) & 0xFF
    expected_response[4] = (cmd_sum >> 8) & 0xFF
    expected_response[5] = cmd_sum & 0xFF  # Lowest byte

    return expected_response


def classify_response(response):
    """Classify a raw serial response.

    The current protocol does not include an explicit type byte. We infer the
    type from the packet shape instead:

    - response beginning with [0, 0] is a debug/hello message
    - response length 6 is a checksum response
    - anything else is unknown / malformed
    """
    response = np.asarray(response, dtype=np.uint8)
    if response.shape[0] >= 2 and response[0] == 0 and response[1] == 0:
        return "debug"
    if response.shape[0] == 6:
        return "checksum"
    return "unknown"


class SerialWorkerError(RuntimeError):
    """A board's serial worker failed in pipelined mode (the original
    error, e.g. a TimeoutError, is chained as __cause__)."""


class PeerWorkerStopped(RuntimeError):
    """A board's serial worker stopped because another board's worker did
    (that worker's own error is the root cause)."""


class ShowMarker:
    """Worker queue item: send SN to every board at time t (a
    time.monotonic() value), or as soon as possible if t is None.

    Each board's worker stops at the marker once it has sent everything
    queued ahead of it (the frame to be shown), and waits until every
    other board's worker has reached it too, so both halves of the display
    update together. The workers record when they arrived, so a late frame
    can be blamed on the board(s) that weren't ready.
    """

    def __init__(self, t, workers):
        self.t = t
        self.workers = workers
        n = len(workers)
        self.ready_at = [None] * n
        self.ready = [threading.Event() for _ in range(n)]
        self.sent = [threading.Event() for _ in range(n)]

    def all_ready(self):
        return all(e.is_set() for e in self.ready)

    def done(self):
        return all(e.is_set() for e in self.sent)


class BoardSerialWorker(threading.Thread):
    """Keep one board's serial traffic ordered while allowing a small
    in-flight window for command pipelining.

    The worker keeps at most ``max_inflight`` commands outstanding. Each
    command is sent immediately while there is room; responses are then read
    and validated in FIFO order as soon as they become available. This keeps
    ordering strict while avoiding the fully serial "send then wait for the
    matching response before sending another" behaviour.

    A ShowMarker in the queue holds everything behind it until the show
    command has been sent (see ShowMarker).

    A response timeout protects against deadlocks where the board stops
    producing replies and the host would otherwise block forever waiting for the
    next byte-to-byte frame. Any error stops the worker and is stored in
    ``last_error`` for the main thread to raise.
    """

    def __init__(
        self,
        ser,
        queue_size=0,
        max_inflight=5,
        response_timeout=2.0,
        index=0,
        name=None,
    ):
        super().__init__(daemon=True, name=name)
        self.ser = ser
        self.queue = Queue(maxsize=queue_size)
        self.max_inflight = max_inflight
        self.response_timeout = response_timeout
        self.index = index
        self._stop_event = threading.Event()
        self._inflight = deque()
        self.last_error = None

    def enqueue(self, cmd):
        cmd = np.asarray(cmd, dtype=np.uint8).copy()
        self.queue.put(cmd)

    def enqueue_marker(self, marker):
        self.queue.put(marker)

    def shutdown(self):
        self._stop_event.set()
        self.queue.put(None)

    def is_idle(self):
        """True once everything queued has been sent and acknowledged."""
        return self.queue.unfinished_tasks == 0 and not self._inflight

    def _validate_response(self, response, expected_response):
        kind = classify_response(response)
        if kind == "checksum" and np.array_equal(response, expected_response):
            logger.debug("Resp rec'd")
            return True
        if kind == "debug":
            logger.debug("Debug msg: %s", bytes(response[2:]).decode())
            return True
        if kind == "unknown":
            logger.warning(
                "Resp invalid, expected %s, got %s",
                expected_response,
                response,
            )
            return False
        logger.warning(
            "Resp invalid, expected %s, got %s",
            expected_response,
            response,
        )
        return False

    def _send(self, cmd):
        send_data_to_arduino(self.ser, cmd)
        self._inflight.append(
            (cmd, calc_expected_response(cmd), time.monotonic())
        )

    def _service_responses(self):
        """Read and validate one response if available. Returns True if
        one was read; raises TimeoutError if the oldest outstanding
        command has waited longer than response_timeout."""
        if not self._inflight:
            return False
        if getattr(self.ser, "in_waiting", 0) > 0:
            response = receive_data_from_arduino(self.ser)
            _, expected_response, _ = self._inflight.popleft()
            self._validate_response(response, expected_response)
            return True
        oldest_cmd, _, sent_at = self._inflight[0]
        if time.monotonic() - sent_at > self.response_timeout:
            raise TimeoutError(
                f"{self.name}: timed out waiting for response to command "
                f"{oldest_cmd!r} after {self.response_timeout:.2f}s"
            )
        return False

    def _show_at_marker(self, marker):
        i = self.index
        marker.ready_at[i] = time.monotonic()
        marker.ready[i].set()

        # Wait for the other board(s) to finish sending this frame.
        while not marker.all_ready():
            if not all(w.is_alive() for w in marker.workers):
                raise PeerWorkerStopped(
                    f"{self.name}: another board's serial worker stopped "
                    "while waiting to show a frame"
                )
            if not self._service_responses():
                time.sleep(0.0005)

        # Wait for the show time. Sleep in short steps (still reading
        # responses) until just before t, then yield-spin: time.sleep(0)
        # releases the GIL so the other worker can wake on time too.
        if marker.t is not None:
            while (remaining := marker.t - time.monotonic()) > 0.002:
                if not self._service_responses():
                    time.sleep(min(remaining - 0.002, 0.0005))
            while time.monotonic() < marker.t:
                time.sleep(0)

        # TODO: Send SN to TEENSY1 only and have it trigger TEENSY2's
        # leds.show() over the GPIO sync wire between the two boards, so
        # both halves update at exactly the same moment. Both workers
        # would still wait at the marker above (TEENSY2's data must be
        # sent before the shared show), but only one would send SN.
        self._send(COMMAND_SN)
        marker.sent[i].set()

        # Report a late frame once (from the first board's worker).
        if i == 0 and marker.t is not None:
            late = max(marker.ready_at) - marker.t
            if late > 0:
                slow = [
                    w.name
                    for w, r in zip(marker.workers, marker.ready_at)
                    if r > marker.t
                ]
                logger.warning(
                    "Frame shown %.1f ms late: %s not ready by show time.",
                    late * 1000,
                    ", ".join(slow),
                )

    def run(self):
        try:
            self._run()
        except Exception as err:
            self.last_error = err
            logger.warning("%s stopped: %s", self.name, err)
            self._stop_event.set()

    def _run(self):
        while True:
            while len(self._inflight) < self.max_inflight:
                try:
                    item = self.queue.get_nowait()
                except Empty:
                    break
                if item is None:
                    self.queue.task_done()
                    self._stop_event.set()
                    break
                if isinstance(item, ShowMarker):
                    self._show_at_marker(item)
                else:
                    self._send(item)
                self.queue.task_done()

            if not self._inflight:
                if self._stop_event.is_set() and self.queue.empty():
                    break
                time.sleep(0.0005)
                continue

            if self._service_responses():
                continue

            if self._stop_event.is_set() and self.queue.empty():
                break

            time.sleep(0.0005)


def _wait_until(t, spin=0.002):
    """Wait until time.monotonic() reaches t, logging a warning if late.

    Sleeps for half the remaining time, repeatedly, until shortly before
    t, then busy-waits the last `spin` seconds. A single long sleep can
    overshoot by several milliseconds (seen on macOS); shorter sleeps
    near t keep any overshoot small.
    """
    late = time.monotonic() - t
    if late > 0:
        logger.warning("Show time missed by %.1f ms.", late * 1000)
        return
    while (remaining := t - time.monotonic()) > spin:
        time.sleep(max((remaining - spin) / 2, 0.0005))
    while time.monotonic() < t:
        pass


def check_response(ser, cmd, timeout_after=1):
    expected_response = calc_expected_response(cmd)
    waiting = True
    timeout_time = time.time() + timeout_after
    while waiting:
        if ser.in_waiting > 0:
            waiting = False
            response = receive_data_from_arduino(ser)
            kind = classify_response(response)
            if kind == "checksum" and np.array_equal(
                response, expected_response
            ):
                logger.debug("Resp rec'd")
            elif kind == "debug":
                logger.debug("Debug msg: %s", bytes(response[2:]).decode())
            else:
                logger.warning(
                    "Resp invalid, expected %s, got %s",
                    expected_response,
                    response,
                )
        if time.time() > timeout_time:
            logger.warning("Timeout")
            break


class Display1593:
    def __init__(
        self,
        ports=SERIAL_PORTS,
        baud_rate=BAUD_RATE,
        number_of_leds=NUMBER_OF_LEDS,
        lock_path=None,
        pipelined=False,
        max_frames_ahead=2,
        max_inflight=5,
        response_timeout=2.0,
    ):
        """
        pipelined: if True, connect() starts the pipeline (see
            start_pipeline()). Otherwise every call is synchronous.
        max_frames_ahead: in pipelined mode, how many show() calls may
            be scheduled but not yet carried out before show() blocks.
        max_inflight: in pipelined mode, how many commands each board
            may have sent but not yet acknowledged.
        response_timeout: in pipelined mode, how long to wait for a
            board's acknowledgement before stopping with an error.
        """
        self.ports = ports
        self.baud_rate = baud_rate
        self._pipelined_on_connect = pipelined
        self.max_frames_ahead = max_frames_ahead
        self.max_inflight = max_inflight
        self.response_timeout = response_timeout
        self._lock = (
            DisplayLock() if lock_path is None else DisplayLock(lock_path)
        )
        self.board_names = list(number_of_leds.keys())
        self.leds_per_board = np.fromiter(
            number_of_leds.values(), dtype="int32"
        )
        self.led_idx = np.concatenate(
            (
                np.zeros(1, dtype=np.int32),
                np.cumsum(self.leds_per_board, dtype=np.int32),
            )
        )
        self.n_leds = self.led_idx[-1]
        self._connections = []
        self.serial_workers = []
        self._reset_pipeline_state()
        self.nearest_neighbours = np.asarray(
            nearest_neighbours, dtype=np.uint16
        )
        self.nearest_neighbour_distances = np.asarray(
            nearest_neighbour_distances, dtype=float
        )
        self.leds = type("LEDLayout", (), {})()
        self.leds.num_cells = int(self.n_leds)
        self.leds.centres_x = np.asarray(centres_x, dtype=float)
        self.leds.centres_y = np.asarray(centres_y, dtype=float)
        self.leds.nearest_neighbours = self.nearest_neighbours
        self.leds.nearest_neighbour_distances = (
            self.nearest_neighbour_distances
        )

    def connect(self, max_attempts=3, lock_timeout=DEFAULT_LOCK_TIMEOUT):
        # Wait here (up to lock_timeout seconds) for exclusive control of
        # the display; raises DisplayLockTimeout if another process is
        # still holding it. Released in disconnect(), or below if
        # connecting fails partway.
        try:
            self._lock.acquire(timeout=lock_timeout)
        except DisplayLockTimeout:
            logger.warning(
                "Could not connect: display is locked by another "
                "process (waited %.1fs).",
                lock_timeout,
            )
            raise
        try:
            connections = {}
            for port in self.ports:
                for attempt in range(1, max_attempts + 1):
                    ser = serial.Serial(port, baudrate=self.baud_rate)
                    _drain_serial_input(ser)
                    # connect_to_arduino() has no checksum on the hello
                    # message it waits for (see check_response() for the
                    # checksummed alternative used elsewhere), so a
                    # corrupted byte can produce a malformed message (an
                    # AssertionError) or a name we don't recognize. Either
                    # way, treat it as a failed attempt and retry rather
                    # than trusting it or crashing.
                    try:
                        status, message = connect_to_arduino(ser)
                    except AssertionError:
                        status, message = 2, "Malformed hello message"
                    if status == 0 and message in self.board_names:
                        logger.info("Connected to port %s.", port)
                        worker_name = message
                        break
                    if status == 0:
                        message = f"unrecognized board name {message!r}"
                    logger.warning(
                        "Attempt %d/%d on port %s failed: %s",
                        attempt,
                        max_attempts,
                        port,
                        message,
                    )
                    ser.close()
                else:
                    logger.error(
                        "Giving up on port %s after %d attempts "
                        "(last error: %s).",
                        port,
                        max_attempts,
                        message,
                    )
                    raise Exception(
                        f"No microcontroller found on port {port} after "
                        f"{max_attempts} attempts (last error: {message})"
                    )
                logger.info("Hello from: %s", worker_name)
                connections[worker_name] = ser

            if set(connections.keys()) != set(self.board_names):
                raise ValueError(
                    "board name mismatch, expected %s, got %s"
                    % (self.board_names, list(connections.keys()))
                )

            # Store connections in same order as expected board names
            self._connections = []
            for name in self.board_names:
                self._connections.append(connections[name])

            self.serial_workers = []
            if self._pipelined_on_connect:
                self.start_pipeline()
        except Exception:
            self._lock.release()
            raise

    @property
    def pipelined(self):
        """True while the pipeline is running (see start_pipeline())."""
        return bool(self.serial_workers)

    def _reset_pipeline_state(self):
        # Commands staged since the last show(), per board, not yet
        # released to the workers.
        self._pending = [[] for _ in self.board_names]
        # True once a frame has been released but not yet shown.
        self._unshown = False
        # ShowMarkers queued but not yet carried out by every worker.
        self._markers = deque()

    def start_pipeline(self, max_inflight=None, response_timeout=None):
        """Switch to pipelined mode: one background thread per board sends
        commands, so building the next frame overlaps with sending the
        current one.

        In pipelined mode the set_*()/clear_all() methods return at once,
        and each show() call shows the frame set *before the previous*
        show() call, then starts sending the frame just set. The first
        frame therefore appears at the second show() call; call flush()
        at the end to show the last one. See README.md.
        """
        if max_inflight is None:
            max_inflight = self.max_inflight
        if response_timeout is None:
            response_timeout = self.response_timeout

        self.stop_pipeline()
        self.serial_workers = [
            BoardSerialWorker(
                ser,
                max_inflight=max_inflight,
                response_timeout=response_timeout,
                index=i,
                name=name,
            )
            for i, (ser, name) in enumerate(
                zip(self._connections, self.board_names)
            )
        ]
        for worker in self.serial_workers:
            worker.start()

    def stop_pipeline(self):
        """Stop the pipeline and return to synchronous mode. Anything
        already released by show() is still sent first; anything set since
        the last show() is discarded (call flush() first to show it)."""
        for worker in self.serial_workers:
            if worker.is_alive():
                worker.shutdown()
        for worker in self.serial_workers:
            if worker.is_alive():
                worker.join(timeout=1)
        self.serial_workers = []
        self._reset_pipeline_state()

    def _stage(self, board_index, cmd):
        self._pending[board_index].append(cmd)

    def _stage_all_boards(self, cmd):
        for board_index in range(len(self.board_names)):
            self._stage(board_index, cmd)

    def _raise_worker_errors(self):
        errors = [w.last_error for w in self.serial_workers if w.last_error]
        # Raise a root cause (e.g. a board that stopped replying) in
        # preference to the knock-on PeerWorkerStopped it causes.
        errors.sort(key=lambda err: isinstance(err, PeerWorkerStopped))
        # A new exception each time: re-raising the worker's own exception
        # object would keep appending to its traceback.
        if errors:
            raise SerialWorkerError(str(errors[0])) from errors[0]
        for worker in self.serial_workers:
            if not worker.is_alive():
                raise SerialWorkerError(f"{worker.name}: worker stopped")

    def _wait_for(self, condition):
        """Poll until condition() is true, raising any worker error."""
        while not condition():
            self._raise_worker_errors()
            time.sleep(0.0005)

    def _queue_show_marker(self, t):
        marker = ShowMarker(t, self.serial_workers)
        for worker in self.serial_workers:
            worker.enqueue_marker(marker)
        self._markers.append(marker)

    def _show_pipelined(self, t):
        self._raise_worker_errors()
        if t is not None and (late := time.monotonic() - t) > 0:
            logger.warning(
                "show() called %.1f ms after its show time.", late * 1000
            )

        # Limit how far the script can get ahead of the display.
        def few_enough_ahead():
            while self._markers and self._markers[0].done():
                self._markers.popleft()
            return len(self._markers) < self.max_frames_ahead

        self._wait_for(few_enough_ahead)

        # Show the frame already released (if any) at t...
        if self._unshown:
            self._queue_show_marker(t)
        # ...then release the frame just set, to be shown next call.
        for worker, cmds in zip(self.serial_workers, self._pending):
            for cmd in cmds:
                worker.queue.put(cmd)
        self._pending = [[] for _ in self.board_names]
        self._unshown = True

    def flush(self, t=None):
        """In pipelined mode, show the last frame and wait until everything
        queued has been sent and acknowledged. Does nothing in synchronous
        mode.

        The last frame is shown at t (a time.monotonic() value), or as
        soon as possible if t is None. If LED values have been set since
        the last show(), the frame before them is shown at t and they are
        shown straight after.
        """
        if not self.pipelined:
            return
        if any(self._pending):
            self.show(t)
            t = None
        if self._unshown:
            self._raise_worker_errors()
            self._queue_show_marker(t)
            self._unshown = False
        self._wait_for(
            lambda: all(worker.is_idle() for worker in self.serial_workers)
        )
        self._markers.clear()

    def check_response(self, ser, cmd, timeout_after=1):
        check_response(ser, cmd, timeout_after=timeout_after)

    def clear_all(self):
        logger.debug("Method clear_all.")
        cmd = COMMAND_LC
        if self.pipelined:
            self._stage_all_boards(cmd)
            return
        for ser in self._connections:
            send_data_to_arduino(ser, cmd)
        for ser in self._connections:
            self.check_response(ser, cmd)

    def set_led(self, i, rgb):
        logger.debug("Method set_led.")
        if i < self.led_idx[0]:
            raise ValueError("invalid led id")
        assert len(rgb) == 3
        if i < self.led_idx[1]:
            led_id = i
            board_index = 0
        elif i < self.led_idx[2]:
            led_id = i - self.led_idx[1]
            board_index = 1
        else:
            raise ValueError("invalid led id")
        # Command L1 - implemented
        cmd = np.array(
            (76, 49, led_id // 256 % 256, led_id % 256, *rgb), dtype=np.uint8
        )
        if self.pipelined:
            self._stage(board_index, cmd)
            return
        ser = self._connections[board_index]
        send_data_to_arduino(ser, cmd)
        self.check_response(ser, cmd)

    def set_leds(self, leds, rgb_array):
        assert rgb_array.shape[1] == 3
        leds = np.array(leds, dtype="int32")
        logger.debug("Method set_leds with %d leds.", leds.shape[0])
        board_leds_0, board_leds_1, rgb_arrays_0, rgb_arrays_1 = (
            _board_leds_with_rgb(leds, rgb_array, self.led_idx)
        )
        board_leds = [board_leds_0, board_leds_1]
        rgb_arrays = [rgb_arrays_0, rgb_arrays_1]
        if self.pipelined:
            for board_index, (leds, rgb_array) in enumerate(
                zip(board_leds, rgb_arrays)
            ):
                n = leds.shape[0]
                if n == 0:
                    continue
                idx = make_idx_array(leds)
                cmd = np.concatenate(
                    [
                        (76, 78, n // 256 % 256, n % 256),
                        np.hstack((idx, rgb_array)).flatten(),
                    ]
                ).astype(np.uint8)
                self._stage(board_index, cmd)
            return

        cmds_sent = {}
        for leds, rgb_array, ser in zip(
            board_leds, rgb_arrays, self._connections
        ):
            n = leds.shape[0]
            if n == 0:
                continue
            idx = make_idx_array(leds)
            # Command LN - implemented
            cmd = np.concatenate(
                [
                    (76, 78, n // 256 % 256, n % 256),
                    np.hstack((idx, rgb_array)).flatten(),
                ]
            ).astype(np.uint8)
            send_data_to_arduino(ser, cmd)
            cmds_sent[ser] = cmd
        for ser, cmd in cmds_sent.items():
            self.check_response(ser, cmd)

    def set_leds_one_colour(self, leds, rgb):
        assert len(rgb) == 3
        leds = np.array(leds, dtype="int32")
        logger.debug(
            "Method set_leds_one_colour with %d leds." % leds.shape[0]
        )
        board_leds_0, board_leds_1 = _board_leds(leds, self.led_idx)
        board_leds = [board_leds_0, board_leds_1]
        if self.pipelined:
            for board_index, leds in enumerate(board_leds):
                n = leds.shape[0]
                if n == 0:
                    continue
                idx = make_idx_array(leds)
                cmd = np.concatenate(
                    [(67, 78, n // 256 % 256, n % 256, *rgb), idx.flatten()]
                ).astype(np.uint8)
                self._stage(board_index, cmd)
            return

        cmds_sent = {}
        for leds, ser in zip(board_leds, self._connections):
            n = leds.shape[0]
            if n == 0:
                continue
            idx = make_idx_array(leds)
            # Command CN - implemented
            cmd = np.concatenate(
                [(67, 78, n // 256 % 256, n % 256, *rgb), idx.flatten()]
            ).astype(np.uint8)
            send_data_to_arduino(ser, cmd)
            cmds_sent[ser] = cmd
        for ser, cmd in cmds_sent.items():
            self.check_response(ser, cmd)

    def set_all_leds(self, rgb_array):
        logger.debug("Method set_all_leds.")
        assert rgb_array.shape == (self.n_leds, 3)
        if self.pipelined:
            for board_index, (i, j) in enumerate(pairwise(self.led_idx)):
                cmd = np.concatenate(
                    [(76, 65), rgb_array[i:j].flatten()]
                ).astype(np.uint8)
                self._stage(board_index, cmd)
            return

        cmds_sent = {}
        for (i, j), ser in zip(pairwise(self.led_idx), self._connections):
            # Command LA - implemented
            cmd = np.concatenate([(76, 65), rgb_array[i:j].flatten()]).astype(
                np.uint8
            )
            send_data_to_arduino(ser, cmd)
            cmds_sent[ser] = cmd
        for ser, cmd in cmds_sent.items():
            self.check_response(ser, cmd)

    def set_all_leds_one_colour(self, rgb):
        logger.debug("Method set_all_leds_one_colour.")
        assert len(rgb) == 3
        # Command CA - implemented
        cmd = np.array((67, 65, *rgb), dtype=np.uint8)
        if self.pipelined:
            self._stage_all_boards(cmd)
            return
        for ser in self._connections:
            send_data_to_arduino(ser, cmd)
        for ser in self._connections:
            self.check_response(ser, cmd)

    def prepare_image(self, image, size=(256, 256)):
        """Crop image to a square and resize it for convert_image()."""
        return _prepare_image(image, size=size)

    def convert_image(self, image_array):
        """Convert a 256x256 RGB image array to 1593 RGB LED intensities."""
        return _convert_image(image_array)

    def show_image(self, filename, dimness=8):
        logger.debug("Method show_image.")
        image = Image.open(filename)
        z = self.convert_image(self.prepare_image(image))
        self.set_all_leds(z**2 / (256 * dimness))

    def show(self, t=None):
        """Show a frame on the display, now or at time t.

        t is a time.monotonic() value; if it has already passed, a
        warning is logged and the frame is shown as soon as possible.

        Synchronous mode: shows the LED values just set. Waits until t
        (if given), then blocks until both boards have acknowledged.

        Pipelined mode: shows the frame set *before the previous* show()
        call - the one already sent to the boards - at t, and starts
        sending the frame just set, to be shown by the next call. Returns
        immediately unless max_frames_ahead show() calls are already
        waiting to be carried out. Raises any error from the background
        threads (e.g. a board that stopped replying).
        """
        logger.debug("Method show.")
        if self.pipelined:
            self._show_pipelined(t)
            return
        if t is not None:
            _wait_until(t)
        # Command SN - implemented.
        # TODO: Send SN to TEENSY1 only and have it trigger TEENSY2's
        # leds.show() over the GPIO sync wire between the two boards, so
        # both halves update at exactly the same moment.
        for ser in self._connections:
            send_data_to_arduino(ser, COMMAND_SN)
        for ser in self._connections:
            self.check_response(ser, COMMAND_SN)

    def disconnect(self):
        if self.pipelined:
            # Show the last frame rather than silently dropping it - unless
            # a board has already failed - but don't let a failure here
            # stop the connections closing.
            if any(w.last_error for w in self.serial_workers):
                logger.warning("Not flushing on disconnect: a board failed.")
            else:
                try:
                    self.flush()
                except Exception:
                    logger.exception("Error flushing pipeline on disconnect.")
        self.stop_pipeline()
        while len(self._connections) > 0:
            ser = self._connections.pop()
            ser.close()
            logger.info("Closed connection to %s.", ser.port)
        self._lock.release()

    def __enter__(self):
        """Enter context manager method"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager method"""
        self.disconnect()
        return False
