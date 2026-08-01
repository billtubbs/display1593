import numpy as np

from schelling import (
    NOMINAL_NEIGHBOUR_DISTANCE,
    Population,
    is_happy,
    is_happy_weighted,
)


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


def test_is_happy_weighted_matches_is_happy_at_nominal_distance():
    n_neighbours = 8
    threshold = 0.35
    distances = np.full(n_neighbours, NOMINAL_NEIGHBOUR_DISTANCE)

    for like_neighbours in range(n_neighbours + 1):
        mask = np.array(
            [True] * like_neighbours + [False] * (n_neighbours - like_neighbours)
        )

        assert is_happy_weighted(mask, distances, threshold) == is_happy(
            like_neighbours, n_neighbours, threshold
        )


def test_is_happy_weighted_can_pass_where_is_happy_fails_for_close_neighbours():
    # 2 of 8 like neighbours: unweighted fraction is 0.25, below threshold.
    threshold = 0.35
    mask = np.array([True, True, False, False, False, False, False, False])
    # The 2 like neighbours are much closer than nominal, so they are
    # weighted more heavily than the 6 unlike neighbours at nominal distance.
    distances = np.array([1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    mean_distance = distances.mean()

    assert mean_distance < NOMINAL_NEIGHBOUR_DISTANCE
    assert not is_happy(2, 8, threshold)
    assert is_happy_weighted(mask, distances, threshold)


def test_is_happy_weighted_can_fail_where_is_happy_passes_for_distant_neighbours():
    # 3 of 8 like neighbours: unweighted fraction is 0.375, above threshold.
    threshold = 0.35
    mask = np.array([True, True, True, False, False, False, False, False])
    # The 3 like neighbours are much farther than nominal, so they are
    # weighted less heavily than the 5 unlike neighbours at nominal distance.
    distances = np.array([20.0, 20.0, 20.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    mean_distance = distances.mean()

    assert mean_distance > NOMINAL_NEIGHBOUR_DISTANCE
    assert is_happy(3, 8, threshold)
    assert not is_happy_weighted(mask, distances, threshold)


def test_population_can_address_second_board_led_ids():
    display = DummyDisplay(n_leds=1593)

    display.set_leds(
        np.array([798], dtype=np.int32),
        np.array([(1, 2, 3)], dtype=np.uint8),
    )

    assert 798 in display.leds
