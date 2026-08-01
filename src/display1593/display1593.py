import logging
import os
import time

from itertools import pairwise


import numpy as np
import serial
from numba import jit, types

from serial_comm import (
    connect_to_arduino,
    receive_data_from_arduino,
    send_data_to_arduino,
)

from display1593.data.ledArray_data_1593 import (
    centres_x,
    centres_y,
    nearest_neighbours,
    nearest_neighbour_distances,
)


# Numba array types
readonly_uint8_array = types.Array(types.uint8, 1, "C", readonly=True)
writable_uint8_array = types.Array(types.uint8, 1, "C")
int32_array = types.Array(types.int32, 1, "C")
int32_array_1d = types.Array(types.int32, 1, "C")
uint8_array_2d = types.Array(types.uint8, 2, "C")


# Set up logging
logger = logging.getLogger(__name__)
LOG_FORMAT = "%(asctime)s.%(msecs)03d|%(levelname)s|%(name)s|%(message)s"
filename = os.path.basename(__file__)
os.path.splitext(os.path.basename(__file__))
logging.basicConfig(
    filename=os.path.splitext(filename)[0] + ".log",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format=LOG_FORMAT,
)

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


class Display1593:
    def __init__(
        self,
        ports=SERIAL_PORTS,
        baud_rate=BAUD_RATE,
        number_of_leds=NUMBER_OF_LEDS,
    ):
        self.ports = ports
        self.baud_rate = baud_rate
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

    def connect(self):
        connections = {}
        for port in self.ports:
            ser = serial.Serial(port, baudrate=self.baud_rate)
            status, message = connect_to_arduino(ser)
            if status == 0:
                logger.info("Connected to port %s.", port)
                worker_name = message
            else:
                logger.debug("Connection to port %s failed.", port)
                raise Exception(message)
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

    def check_response(self, ser, cmd, timeout_after=1):
        expected_response = calc_expected_response(cmd)
        waiting = True
        timeout_time = time.time() + timeout_after
        while waiting:
            if ser.in_waiting > 0:
                waiting = False
                response = receive_data_from_arduino(ser)
                if np.array_equal(response, expected_response):
                    logger.debug("Resp rec'd")
                elif np.array_equal(response[:2], [0, 0]):
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

    def clear_all(self):
        logger.debug("Method clear_all.")
        cmd = COMMAND_LC
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
            ser = self._connections[0]
        elif i < self.led_idx[2]:
            led_id = i - self.led_idx[1]
            ser = self._connections[1]
        else:
            raise ValueError("invalid led id")
        # Command L1 - implemented
        cmd = np.array(
            (76, 49, led_id // 256 % 256, led_id % 256, *rgb), dtype=np.uint8
        )
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
        for ser in self._connections:
            send_data_to_arduino(ser, cmd)
        for ser in self._connections:
            self.check_response(ser, cmd)

    def show_now(self):
        logger.debug("Method show_now.")
        # Command SN - implemented
        # TODO: In future this will be synchronized by comms between boards
        cmd = COMMAND_SN
        for ser in self._connections:
            send_data_to_arduino(ser, cmd)
        for ser in self._connections:
            self.check_response(ser, cmd)

    def disconnect(self):
        while len(self._connections) > 0:
            ser = self._connections.pop()
            ser.close()
            logger.info("Closed connection to %s.", ser.port)

    def __enter__(self):
        """Enter context manager method"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager method"""
        self.disconnect()
        return False
