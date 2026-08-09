import time
from itertools import cycle
from pathlib import Path

import numpy as np
from display1593 import Display1593


DATA_DIR = Path(__file__).parent / "data"
TIME_STEP = 0.0625  # seconds


def load_led_frames(data_dir):
    """Loads precomputed LED RGB frames (one CSV per frame) from data_dir."""
    frame_paths = sorted(Path(data_dir).glob("*.csv"))
    return [np.loadtxt(p, delimiter=",", dtype="uint8") for p in frame_paths]


def main(dis, img_data):

    dis.clear_all()

    print("Starting...")
    img_cycle = cycle(img_data)

    # Prepare to log timing
    start_time = time.monotonic()
    scheduled_times = []
    actual_times = []
    wait_times = []

    next_time = time.monotonic()
    try:
        while True:
            z = next(img_cycle)
            dis.set_all_leds(z)

            next_time += TIME_STEP
            scheduled_times.append(next_time)

            # Synchronize display to clock
            wait_time = max(0, next_time - time.monotonic())
            wait_times.append(wait_time)
            time.sleep(wait_time)
            dis.show_now()

            actual_times.append(time.monotonic())

    except KeyboardInterrupt:
        print("Stopped.")

    # Convert to elapsed time since start
    scheduled_times = np.array(scheduled_times) - start_time
    actual_times = np.array(actual_times) - start_time

    for sch, act, wait in zip(scheduled_times, actual_times, wait_times):
        print(f"{sch:6.3f} {act:6.3f} {wait * 1000:6.2f} ms")


if __name__ == "__main__":
    img_data = load_led_frames(DATA_DIR)
    with Display1593() as dis:
        main(dis, img_data)
