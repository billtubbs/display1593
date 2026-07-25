from itertools import cycle
from collections import deque
import numpy as np
import serial
from serial_comm import (
    connect_to_arduino,
    send_data_to_arduino,
    receive_data_from_arduino,
)
from display1593 import *
import logging
import os
import time


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


def move_pointer_commands(i1, i2):
    return [
        np.array(list(b"L1") + [0, i1, 0, 0, 0], dtype="uint8"),
        np.array(list(b"L1") + [0, i2, 20, 32, 0], dtype="uint8"),
    ]


def manual_testing(ser):

    import pygame

    status, message = connect_to_arduino(ser)
    if status == 0:
        worker_name = message
    else:
        raise Exception(message)
    logger.info(f"Hello from: {worker_name}")

    # # Clear display
    # data = np.array(list(b'LC'), dtype="uint8")
    # send_data_to_arduino(ser, data)

    # for i in range(50):
    #     data = np.array(list(b'L1') + [0, i, 32, 4, 32], dtype="uint8")
    #     send_data_to_arduino(ser, data)

    # # Show LED updates now
    # np.array(list(b'SN'), dtype="uint8")
    # send_data_to_arduino(ser, data)

    waiting_for_response = False
    command_queue = deque()
    led_id = 0
    while True:
        # Check for new key presses
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    led_id_new = (led_id - 1) % 100
                    command_queue.extendleft(
                        move_pointer_commands(led_id, led_id_new)
                    )
                    led_id = led_id_new
                    logger.info("LEFT key pressed.")
                if event.key == pygame.K_RIGHT:
                    led_id_new = (led_id + 1) % 100
                    command_queue.extendleft(
                        move_pointer_commands(led_id, led_id_new)
                    )
                    led_id = led_id_new
                    logger.info("RIGHT key pressed.")

        if ser.in_waiting == 0 and waiting_for_response is False:
            if len(command_queue) > 0:
                data = command_queue.pop()
                send_data_to_arduino(ser, data)
                # logger.info("Command sent.")
                waiting_for_response = True

        if ser.in_waiting > 0:
            logger.info("Receiving data...")
            data_received = receive_data_from_arduino(ser)
            logger.info("Data received.")
            if np.array_equal(data_received[:2], [0, 0]):
                # Debug message from Arduino
                logger.info(f"Debug message: {data_received[2:].tobytes()}")
            else:
                # Process data from Arduino - data integrity checks
                assert data_received.shape[0] == 6
                num_bytes_received = int(data_received[0]) * 256 + int(
                    data_received[1]
                )
                data_sum = (
                    int(data_received[2]) * 16777216
                    + int(data_received[3]) * 65536
                    + int(data_received[4]) * 256
                    + int(data_received[5])
                )
                assert num_bytes_received == data.shape[0]
                if data_sum == data.sum():
                    pass
                    # logger.info(f"Test {i_iter} complete.")
                else:
                    logger.debug(f"Test {i_iter} checksum failed.")
                    logger.debug(
                        f"Received: {data_sum}, expected: {data.sum()}."
                    )
                i_iter += 1
                waiting_for_response = False
                time.sleep(0.5)


def run_test(board, ser):

    logger.info("Preparing test data...")

    command_list = []
    command_list.append(clear_all_leds())

    command_list.append(show_now())

    # LEDs per strip
    leds_per_strip = {
        "TEENSY1": [100, 100, 98, 100, 100, 100, 100, 100],
        "TEENSY2": [99, 99, 99, 100, 100, 100, 100, 98],
    }
    first_led_of_strip = {
        name: np.cumsum([0] + leds) for name, leds in leds_per_strip.items()
    }
    leds_per_strip = {
        name: np.array(leds) for name, leds in leds_per_strip.items()
    }
    MAX_LEDS_PER_STRIP = 100

    # Iterate over strips
    for s in range(0, 8):
        start_led = s * MAX_LEDS_PER_STRIP
        n_leds = leds_per_strip[board][s]

        # Light first LED
        command_list.append(set_led(start_led, GREEN))

        # Light last LED
        command_list.append(set_led(start_led + n_leds - 1, RED))

        # Light all in between leds
        leds = np.arange(start_led + 1, start_led + n_leds - 1)
        #         rgb_array = 16 * np.ones((leds.shape[0], 3), dtype='uint8')
        #         command_list.append(set_leds(leds, rgb_array))
        rgb = np.full((3,), 16, dtype=np.uint8)
        command_list.append(set_leds_one_colour(leds, rgb))

        # Try to light additional leds at end of strip that should
        # not exist
        if n_leds < MAX_LEDS_PER_STRIP:
            leds = np.arange(
                start_led + n_leds, start_led + MAX_LEDS_PER_STRIP
            )
            rgb_array = np.repeat([YELLOW], leds.shape[0], axis=0)
            command_list.append(set_leds(leds, rgb_array))

        command_list.append(show_now())

    command_list.append(show_now())

    # Iterator to cycle through test data
    command_list_cycle = cycle(command_list)

    t_start = time.time()
    logger.info("Test start")

    waiting_for_response = False
    n_iter = len(command_list)
    i_iter = 0
    while i_iter < n_iter:
        if ser.in_waiting == 0 and waiting_for_response is False:
            try:
                data = next(command_list_cycle)
            except IndexError:
                break
            # logger.info(f"Sending Test {i_iter} data...")
            send_data_to_arduino(ser, data)
            # logger.info("Data sent.")
            waiting_for_response = True

        if ser.in_waiting > 0:
            # logger.info("Receiving data...")
            data_received = receive_data_from_arduino(ser)
            # logger.info("Data received.")
            if np.array_equal(data_received[:2], [0, 0]):
                # Debug message from Arduino
                logger.info(f"Debug message: {data_received[2:].tobytes()}")
                pass
            else:
                # Process data from Arduino - data integrity checks
                assert data_received.shape[0] == 6
                num_bytes_received = int(data_received[0]) * 256 + int(
                    data_received[1]
                )
                data_sum = (
                    int(data_received[2]) * 16777216
                    + int(data_received[3]) * 65536
                    + int(data_received[4]) * 256
                    + int(data_received[5])
                )
                assert num_bytes_received == data.shape[0]
                assert data_sum == data.sum()
                # logger.info(f"Test {i_iter} complete.")
                i_iter += 1
                waiting_for_response = False

    t_end = time.time()
    logger.info(f"Elapsed time: {t_end - t_start:.3f}s for {n_iter} tests.")
    logger.info(f"Cycle time: {(t_end - t_start) * 1000 / n_iter:.0f}ms.")


def open_serial_connections(ports):
    connections = {}
    for port in SERIAL_PORTS:
        conn = serial.Serial(port, baudrate=BAUD_RATE)
        logger.info(f"Connected to port {port}.")

        status, message = connect_to_arduino(conn)
        if status == 0:
            worker_name = message
        else:
            raise Exception(message)
        logger.info(f"Hello from: {worker_name}")
        connections[worker_name] = conn

    return connections


def close_serial_connections(connections):
    for name, conn in connections.items():
        conn.close()
        logger.info(f"Closed connection to {name}.")


def main():
    logger.info("=" * 35)
    logger.info(f"{filename} started.")
    connections = open_serial_connections(SERIAL_PORTS)
    for board in ["TEENSY1", "TEENSY2"]:
        run_test(board, connections[board])

    # manual_testing(connections['TEENSY1'])
    close_serial_connections(connections)
    logger.info(f"{filename} ended.")


if __name__ == "__main__":
    main()
