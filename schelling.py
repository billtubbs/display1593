#!/usr/bin/python
"""Run a Schelling segregation simulation on the 1593-LED display.

The script keeps the original simulation behaviour, but it uses the newer
Display1593 API rather than the legacy display wrapper.  Because the newer
API does not expose the older LED geometry helpers, the model uses a simple
2D coordinate map derived from the LED index for neighbour calculations.
"""

import logging
import logging.handlers
import sys
import time
from pathlib import Path
from random import shuffle

import numpy as np
from scipy.spatial import KDTree

from display1593 import Display1593

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

COLOURS = [
    (46, 23, 0),
    (0, 46, 0),
    (23, 23, 23),
    (0, 0, 46),
    (29, 15, 8),
    (46, 46, 0),
    (25, 0, 0),
]

logger = logging.getLogger(__name__)
LOG_PATH = BASE_DIR / "schelling.log"
LOG_FORMAT = "%(asctime)s.%(msecs)03d|%(levelname)s|%(name)s|%(message)s"

if not logger.handlers:
    handler = logging.handlers.RotatingFileHandler(
        LOG_PATH,
        maxBytes=256 * 1024,
        backupCount=2,
    )
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format=LOG_FORMAT,
)


class Agent:
    """Represent one agent in the Schelling segregation model."""

    def __init__(self, population, group, threshold, id_):

        self.population = population
        self.group = group
        self.threshold = threshold
        self.colour = self.population.colours[group]
        self.id = int(id_)
        self.location = (
            self.population.coords_x[self.id],
            self.population.coords_y[self.id],
        )
        self.n_neighbours = self.population.n_neighbours
        self.like_neighbours = 0
        self.neighbour_ids = []

    def happy(self):
        """Return whether the agent is happy with its current neighbours."""

        h = (float(self.like_neighbours) / self.n_neighbours) >= self.threshold
        return h

    def move(self, show=True):
        """Move the agent to a random empty location."""

        # TODO: Could make this more systematic?
        # Carry out looped search here maybe
        new_id = int(np.random.choice(self.population.empty_spaces))
        self.population.empty_spaces.append(self.id)
        self.unshow()
        self.id = new_id
        self.population.empty_spaces.remove(self.id)

        self.location = (
            self.population.coords_x[self.id],
            self.population.coords_y[self.id],
        )
        if show:
            self.show()

    def show(self):
        """Show the agent on the LED array by lighting the
        appropriate LED with the agent's group colour.
        """

        self.population.display.set_leds(
            np.array([self.id], dtype=np.int32),
            np.array([self.colour], dtype=np.uint8),
        )

    def unshow(self):
        """Clear the LED representing the agent."""

        self.population.display.set_leds(
            np.array([self.id], dtype=np.int32),
            np.array([self.population.background_col], dtype=np.uint8),
        )
        self.population.display.show_now()


class Population:
    """Manage the agents and their updates for one simulation run."""

    def __init__(
        self,
        dis,
        n,
        probs,
        thresholds,
        n_neighbours=9,
        cols=None,
        background_col=(0, 0, 0),
    ):

        self.display = dis
        self.n_agents = n
        self.probs = probs
        self.n_groups = len(probs)
        if cols is None:
            cols = COLOURS[: self.n_groups]
        self.colours = cols
        self.background_col = background_col
        self.agents = []
        self.n_neighbours = n_neighbours
        self.n_cells = int(getattr(dis, "n_leds", 0))

        layout = getattr(dis, "leds", None)
        centres_x = getattr(layout, "centres_x", None)
        centres_y = getattr(layout, "centres_y", None)

        if centres_x is None or centres_y is None:
            raise ValueError(
                "Display object must provide real LED layout coordinates"
            )

        coords_count = min(self.n_cells, len(centres_x), len(centres_y))
        self.coords_x = np.asarray(centres_x[:coords_count], dtype=float)
        self.coords_y = np.asarray(centres_y[:coords_count], dtype=float)

        self.empty_spaces = list(range(self.n_cells))

        groups = np.random.choice(self.n_groups, p=probs, size=n)
        agent_ids = np.random.choice(self.empty_spaces, size=n, replace=False)
        self.empty_spaces = list(
            set(self.empty_spaces) - {int(i) for i in agent_ids}
        )

        self.agents = [
            Agent(self, group, thresholds[group], agent_id)
            for group, agent_id in zip(groups, agent_ids)
        ]

        self.last_agent = -1

    def count_like_neighbours(self, agent):

        assert len(agent.neighbour_ids) == self.n_neighbours
        tally = dict.fromkeys(range(self.n_groups), 0)

        for n in agent.neighbour_ids:
            n_grp = self.agents[n].group
            tally[n_grp] = tally.get(n_grp, 0) + 1

        agent.like_neighbours = tally[agent.group]

    def update_all_agents(self):
        """Advance the population by one update round.

        Returns True if any agent moved during the round, otherwise False.
        """

        # build a list of all agent locations
        # This would be faster if they were already in one array
        all_locations = [agent.location for agent in self.agents]

        # Flag used to detect when no agents moved in one round
        any_moved = False
        moved = True

        logger.info("Updating all agents...")
        for i, agent in enumerate(self.agents):
            # logging.info("Checking neighbours for agent %d group: %d",
            #             i, agent.group)

            # Build a KDTree from all agent locations
            if moved:
                tree = KDTree(all_locations)
                # logging.info("KD-Tree rebuilt")
            moved = False

            # Query the KDTree to find the k nearest neighbours.
            # KDTree.query returns two arrays, the first contains the
            # nearest neighbour distances, the second contains the
            # indeces of the nearest neighbours. Here, we ignore the
            # first row as this is the location of the current agent.
            k = self.n_neighbours + 1
            agent.neighbour_ids = tree.query(agent.location, k=k)[1][1:]

            # logging.info("Agent's neighbours: %s", str(agent.neighbour_ids))
            self.count_like_neighbours(agent)
            # logging.info("Agent has %d like neighbours.",
            #             agent.like_neighbours)

            if not agent.happy():
                # logging.info("Agent not happy...")
                # Now rebuild the KDTree from all agent locations except
                # the current agent's location (this makes hunting for
                # a new location faster)
                del all_locations[i]
                tree = KDTree(all_locations)

                searches = 0
                while not agent.happy():
                    agent.move(show=False)
                    moved = True
                    any_moved = True

                    k = self.n_neighbours
                    agent.neighbour_ids = tree.query(agent.location, k=k)[1]

                    # Because the current agent's location was not in the list
                    # of points provided to KDTree, need to increment all
                    # indeces > i by 1
                    for j, neighbour_id in enumerate(agent.neighbour_ids):
                        if neighbour_id > i:
                            agent.neighbour_ids[j] += 1

                    # logging.info("Agent's neighbours: %s",
                    #             str(agent.neighbour_ids))

                    self.count_like_neighbours(agent)

                    # logging.info("Agent has %d like neighbours.",
                    #              agent.like_neighbours)
                    searches += 1
                    if searches > 50:
                        # logging.info("Gave up looking.")
                        break

                # Put the current agent's location back in the list
                all_locations.insert(i, agent.location)

            if moved == True:
                agent.show()
                logger.info("Agent %d moved.", i)

        return any_moved

    def move_next_agent(self):
        """Find the next agent, after the last one moved, that wants
        to move, move it to a new location, and return it.

        Agents are checked in order starting after ``last_agent``,
        wrapping around the population. Returns None if no agent
        wants to move.
        """

        n = len(self.agents)
        all_locations = [agent.location for agent in self.agents]
        tree = KDTree(all_locations)
        k = self.n_neighbours + 1

        for step in range(n):
            i = (self.last_agent + 1 + step) % n
            agent = self.agents[i]
            agent.neighbour_ids = tree.query(agent.location, k=k)[1][1:]
            self.count_like_neighbours(agent)

            if agent.happy():
                continue

            del all_locations[i]
            move_tree = KDTree(all_locations)

            searches = 0
            while not agent.happy():
                agent.move(show=False)

                agent.neighbour_ids = move_tree.query(
                    agent.location, k=self.n_neighbours
                )[1]
                for j, neighbour_id in enumerate(agent.neighbour_ids):
                    if neighbour_id > i:
                        agent.neighbour_ids[j] += 1

                self.count_like_neighbours(agent)
                searches += 1
                if searches > 50:
                    break

            agent.show()
            logger.info("Agent %d moved.", i)
            self.last_agent = i
            return agent

        return None

    def show(self):
        """Show all agents on the LED array."""

        for agent in self.agents:
            agent.show()

        if self.empty_spaces:
            empty_ids = np.array(self.empty_spaces, dtype=np.int32)
            empty_rgb = np.tile(
                np.array([self.background_col], dtype=np.uint8),
                (len(empty_ids), 1),
            )
            self.display.set_leds(empty_ids, empty_rgb)

        self.display.show_now()

    def debug_led_range(self):
        """Log the LED range the script will target."""

        logger.info(
            "LED range: 0..%d (n_leds=%d, empty spaces=%d)",
            self.n_cells - 1,
            self.n_cells,
            len(self.empty_spaces),
        )

    def unshow(self):
        """Clear all agents on the LED array."""

        for agent in self.agents:
            agent.unshow()

        self.display.show_now()


def main():
    """Run the Schelling simulation loop until interrupted."""

    logger.info("\n\n------- Schelling Segregation Model Simulation -------\n")

    dis = Display1593()
    dis.connect()
    try:
        cols = list(COLOURS)

        while True:
            logger.info("Initializing population model...")

            # Randomly assign population and model parameters
            # Number of population groups
            p = [0.5, 0.4, 0.1]
            n_groups = np.random.choice(range(2, 5), p=p)

            # Number of neighbours in happiness calculation
            n_neighbours = 9

            # Happiness thresholds
            thresholds = np.random.choice([0.25, 0.35, 0.5], size=n_groups)

            # Number of agents
            n_agents = dis.n_leds - (100 + n_groups * 100)

            x = [(np.random.rand() + 0.25) for i in range(n_groups)]
            t = sum(x)
            probs = [p / t for p in x]

            # Randomly sort the colours
            shuffle(cols)

            population = Population(
                dis,
                n_agents,
                probs,
                thresholds,
                n_neighbours=n_neighbours,
                cols=cols[0:n_groups],
            )

            logger.info("%d agents initialized.", n_agents)
            logger.info("%d population groups.", population.n_groups)
            logger.info("Distribution: %s", str(population.probs))
            logger.info("Thresholds: %s", str(thresholds.tolist()))
            logger.info("Number of nearest neighbours: %d", n_neighbours)

            logger.info("Displaying initial population...")
            dis.clear_all()
            population.show()

            logger.info("Model updating started...")
            update_interval = 1.0
            while True:
                tick_start = time.monotonic()
                if population.move_next_agent() is None:
                    break
                elapsed = time.monotonic() - tick_start
                if elapsed < update_interval:
                    time.sleep(update_interval - elapsed)

            logger.info("Stable population reached.")
            d = 2
            logger.info("Waiting %d mins...", d)
            time.sleep(d * 60)
    finally:
        dis.disconnect()

    logger.info("Results")
    logger.info("   #:   id,  g,       x,       y, nn, neighbour_ids")
    for i, agent in enumerate(population.agents):
        logger.info(
            "%4d: %4d, %2d, %7.2f, %7.2f, %2d, %s",
            i,
            agent.id,
            agent.group,
            agent.location[0],
            agent.location[1],
            agent.like_neighbours,
            str(agent.neighbour_ids),
        )


if __name__ == "__main__":
    main()
