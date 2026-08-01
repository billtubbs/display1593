import numpy as np

from schelling import Population


class DummyDisplay:
    def __init__(self, n_leds=4):
        self.n_leds = n_leds
        self.leds = {}
        self.show_now_calls = 0

    def set_led(self, i, rgb):
        self.leds[i] = rgb

    def set_leds(self, ids, rgb_array):
        for i, rgb in zip(ids, rgb_array):
            self.leds[int(i)] = tuple(rgb)

    def clear_all(self):
        self.leds.clear()

    def show_now(self):
        self.show_now_calls += 1


def test_population_show_flushes_display():
    display = DummyDisplay()
    population = Population(display, 1, [1.0], [0.0], n_neighbours=1)

    population.show()

    assert display.show_now_calls == 1


def test_agent_move_flushes_display():
    display = DummyDisplay()
    population = Population(display, 1, [1.0], [0.0], n_neighbours=1)
    agent = population.agents[0]

    agent.move(show=True)

    assert display.show_now_calls >= 1


def test_population_uses_full_led_range():
    display = DummyDisplay(n_leds=1593)
    population = Population(display, 1, [1.0], [0.0], n_neighbours=1)

    assert population.n_cells == 1593
    assert len(population.empty_spaces) == 1593


def test_population_can_address_second_board_led_ids():
    display = DummyDisplay(n_leds=1593)

    display.set_leds(
        np.array([798], dtype=np.int32),
        np.array([(1, 2, 3)], dtype=np.uint8),
    )

    assert 798 in display.leds
