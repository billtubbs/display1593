import time
import os

import numpy as np
from itertools import cycle
from display1593 import Display1593
from PIL import Image


dimness = 6

FILENAMES = ["fire1.png", "fire2.png", "fire3.png"]
IMAGE_DIR = "images"
TIME_STEP = 0.0625  # seconds


def main(dis, filenames):

    dis.clear_all()
    n_images = len(filenames)

    img_data = []
    for filename in filenames:
        img = Image.open(os.path.join(IMAGE_DIR, filename))
        data = dis.prepare_image(np.array(img)[:, :, :3])
        z = dis.convert_image(data) ** 2 / (256 * dimness)
        img_data.append(z.astype("uint8"))

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
    with Display1593() as dis:
        main(dis, FILENAMES)
