import unittest

from schelling import Population


class DummyDisplay:
    def __init__(self, n_leds=4):
        self.n_leds = n_leds
        self.leds = {}
        self.show_now_calls = 0

    def set_led(self, i, rgb):
        self.leds[i] = rgb

    def clear_all(self):
        self.leds.clear()

    def show_now(self):
        self.show_now_calls += 1


class SchellingDisplayTests(unittest.TestCase):
    def test_population_show_flushes_display(self):
        display = DummyDisplay()
        population = Population(display, 1, [1.0], [0.0], n_neighbours=1)

        population.show()

        self.assertEqual(display.show_now_calls, 1)

    def test_agent_move_flushes_display(self):
        display = DummyDisplay()
        population = Population(display, 1, [1.0], [0.0], n_neighbours=1)
        agent = population.agents[0]

        agent.move(show=True)

        self.assertGreaterEqual(display.show_now_calls, 1)


if __name__ == "__main__":
    unittest.main()
