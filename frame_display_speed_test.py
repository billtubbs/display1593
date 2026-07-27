# Simple test script

import time
import numpy as np

from display1593 import Display1593

dis = Display1593()
dis.connect()

# Prepare LED data
N_FRAMES = 20
data = []
for col in range(N_FRAMES):
    data.append(np.full((1593, 3), col, dtype="uint8"))

FRAME_PERIOD = 0.050  # 50 ms (20 Hz)

next_frame_time = time.perf_counter()

try:
    while True:
        # Count up
        for d in data:
            dis.set_all_leds(d)
            dis.show_now()

            next_frame_time += FRAME_PERIOD
            delay = next_frame_time - time.perf_counter()
            if delay > 0:
                time.sleep(delay)

        # Count down (skip first and last frames)
        for d in data[-2:0:-1]:
            dis.set_all_leds(d)
            dis.show_now()

            next_frame_time += FRAME_PERIOD
            delay = next_frame_time - time.perf_counter()
            if delay > 0:
                time.sleep(delay)

except KeyboardInterrupt:
    pass

dis.clear_all()
dis.show_now()
