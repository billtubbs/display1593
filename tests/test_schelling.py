from types import SimpleNamespace

import numpy as np
import pytest

from schelling import (
    Population,
    is_happy,
    is_happy_weighted,
)


def _make_nearest_neighbours(n_leds, width=18):
    """Build a simple, valid (but not geometrically meaningful)
    nearest_neighbours/nearest_neighbour_distances pair for testing:
    the neighbours of led i are i+1, i-1, i+2, i-2, ... (mod n_leds)."""

    width = min(width, n_leds - 1)
    offsets = []
    k = 1
    while len(offsets) < width:
        offsets.append(k)
        if len(offsets) < width:
            offsets.append(-k)
        k += 1
    offsets = np.array(offsets[:width])
    ids = (np.arange(n_leds)[:, None] + offsets) % n_leds
    distances = np.tile(np.abs(offsets).astype(float), (n_leds, 1))
    return ids.astype(int), distances


class DummyDisplay:
    def __init__(self, n_leds=4):
        self.n_leds = n_leds
        nearest_neighbours, nearest_neighbour_distances = (
            _make_nearest_neighbours(n_leds)
        )
        # Matches the real Display1593 API: `.leds` exposes LED layout
        # coordinates and the nearest_neighbours table, not per-LED
        # colours (see display1593.py).
        self.leds = SimpleNamespace(
            centres_x=np.arange(n_leds, dtype=float),
            centres_y=np.zeros(n_leds, dtype=float),
            nearest_neighbours=nearest_neighbours,
            nearest_neighbour_distances=nearest_neighbour_distances,
        )
        self.led_colours = {}
        self.show_now_calls = 0

    def set_led(self, i, rgb):
        self.led_colours[i] = rgb

    def set_leds(self, ids, rgb_array):
        for i, rgb in zip(ids, rgb_array):
            self.led_colours[int(i)] = tuple(rgb)

    def clear_all(self):
        self.led_colours.clear()

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
    assert len(population.empty_spaces) == 1593 - 1


def test_is_happy_weighted_matches_is_happy_at_equal_distances():
    n_neighbours = 8
    threshold = 0.35
    distances = np.full(n_neighbours, 50.0)

    for like_neighbours in range(n_neighbours + 1):
        mask = np.array(
            [True] * like_neighbours
            + [False] * (n_neighbours - like_neighbours)
        )

        assert is_happy_weighted(mask, distances, threshold) == is_happy(
            like_neighbours, n_neighbours, threshold
        )


def test_is_happy_weighted_can_pass_where_is_happy_fails_for_close_neighbours():
    # 2 of 8 like neighbours: unweighted fraction is 0.25, below threshold.
    threshold = 0.35
    mask = np.array([True, True, False, False, False, False, False, False])
    # The 2 like neighbours are much closer than the other 6, so they are
    # weighted more heavily.
    near, far = 5.0, 50.0
    distances = np.array([near, near, far, far, far, far, far, far])

    assert not is_happy(2, 8, threshold)
    assert is_happy_weighted(mask, distances, threshold)


def test_is_happy_weighted_can_fail_where_is_happy_passes_for_distant_neighbours():
    # 3 of 8 like neighbours: unweighted fraction is 0.375, above threshold.
    threshold = 0.35
    mask = np.array([True, True, True, False, False, False, False, False])
    # The 3 like neighbours are much farther than the other 5, so they are
    # weighted less heavily.
    far, near = 200.0, 50.0
    distances = np.array([far, far, far, near, near, near, near, near])

    assert is_happy(3, 8, threshold)
    assert not is_happy_weighted(mask, distances, threshold)


def test_is_happy_weighted_with_no_occupied_neighbours_is_happy():
    # An agent with no occupied neighbour cells has nothing to be
    # unhappy about.
    assert is_happy_weighted(
        np.array([], dtype=bool), np.array([], dtype=float), 0.9
    )


def test_update_agent_neighbours_only_includes_occupied_cells():
    display = DummyDisplay(n_leds=20)
    population = Population(display, 8, [1.0], [0.0], n_neighbours=6)
    agent = population.agents[0]

    population.update_agent_neighbours(agent)

    candidates = population.nearest_neighbours[agent.id, :6]
    expected_ids = candidates[population.group_at[candidates] >= 0]

    assert set(agent.neighbour_ids.tolist()) == set(expected_ids.tolist())
    assert len(agent.neighbour_ids) <= 6
    assert np.all(population.group_at[agent.neighbour_ids] >= 0)


def test_group_at_updated_when_agent_moves():
    display = DummyDisplay(n_leds=20)
    population = Population(display, 5, [1.0], [0.0], n_neighbours=4)
    agent = population.agents[0]
    old_id = agent.id

    agent.move(show=False)

    assert population.group_at[old_id] == -1
    assert population.group_at[agent.id] == agent.group


def test_population_rejects_n_neighbours_exceeding_table_width():
    display = DummyDisplay(n_leds=6)  # table width = min(18, 5) = 5

    with pytest.raises(ValueError):
        Population(display, 1, [1.0], [0.0], n_neighbours=6)


def test_population_can_address_second_board_led_ids():
    display = DummyDisplay(n_leds=1593)

    display.set_leds(
        np.array([798], dtype=np.int32),
        np.array([(1, 2, 3)], dtype=np.uint8),
    )

    assert 798 in display.led_colours
