"""Interactive script to visually check the static nearest_neighbours
data against the real LED display.

Lights up one LED at a time in white, together with its 6 nearest
neighbours (from the static nearest_neighbours array) in red, so you
can step through the array and check by eye whether the listed
neighbours are actually next to the lit LED.

Controls (press Enter to submit):
    <Enter>  - go to the next LED
    b        - go back to the previous LED
    <number> - jump directly to that LED id
    q        - quit
"""

from display1593 import Display1593
from display1593.data.ledArray_data_1593 import nearest_neighbours, num_cells

CENTRE_COLOUR = (32, 32, 32)  # white
NEIGHBOUR_COLOUR = (32, 0, 0)  # red


def show_led_and_neighbours(dis, led_id):
    neighbours = nearest_neighbours[led_id]
    dis.clear_all()
    dis.set_leds_one_colour(neighbours, NEIGHBOUR_COLOUR)
    dis.set_led(led_id, CENTRE_COLOUR)
    dis.show_now()
    print(f"LED {led_id} (white), neighbours {neighbours.tolist()} (red)")


def main():
    with Display1593() as dis:
        led_id = 0
        show_led_and_neighbours(dis, led_id)
        print(__doc__)
        while True:
            cmd = input(f"[led {led_id}] > ").strip().lower()
            if cmd == "q":
                break
            elif cmd == "":
                led_id = (led_id + 1) % num_cells
            elif cmd == "b":
                led_id = (led_id - 1) % num_cells
            elif cmd.isdigit() and 0 <= int(cmd) < num_cells:
                led_id = int(cmd)
            else:
                print(f"Unrecognized input. Enter a number from 0 to {num_cells - 1}, 'b', or 'q'.")
                continue
            show_led_and_neighbours(dis, led_id)

        dis.clear_all()
        dis.show_now()


if __name__ == "__main__":
    main()
