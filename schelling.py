#!/usr/bin/python
"""Run a Schelling segregation simulation on the 1593-LED display.

Each agent lives at a specific LED cell. Neighbour look-ups use the
display's precomputed nearest_neighbours table (a periodic KDTree over
the real LED layout, see ledArray_data_1593.py), rather than building a
KDTree over agent positions at runtime - this is both faster and models
the true (wrap-around) LED adjacency.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from random import shuffle

import numpy as np

from display1593 import Display1593
from display1593.logging_utils import configure_root_logging

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

COLOURS = [
    (46, 23, 0),
    (9, 35, 7),
    (23, 23, 23),
    (0, 0, 38),
    (29, 15, 8),
    (34, 34, 0),
    (25, 0, 0),
]

# Happiness threshold values to sample from (one is picked per group,
# as the mean of that group's per-agent threshold distribution - see
# THRESHOLD_STD and sample_thresholds())
# THRESHOLD_VALUES = [0.25, 0.35, 0.5]
THRESHOLD_VALUES = [0.5]

# Standard deviation of each group's per-agent threshold distribution
THRESHOLD_STD = 0.1

logger = logging.getLogger(__name__)
LOG_PATH = BASE_DIR / "schelling.log"

configure_root_logging(LOG_PATH)


def beta_params(mean, std):
    """Convert a target mean and standard deviation (both in (0, 1)) to
    Beta distribution shape parameters (alpha, beta).

    The maximum possible standard deviation for a distribution on
    [0, 1] with a given mean is sqrt(mean * (1 - mean)) (the std of a
    two-point distribution sitting at 0 and 1) - raises ValueError if
    std meets or exceeds that.
    """
    max_variance = mean * (1 - mean)
    variance = std ** 2
    if variance >= max_variance:
        raise ValueError(
            f"std={std} too large for mean={mean}; "
            f"max is {max_variance ** 0.5:.4f}"
        )
    nu = max_variance / variance - 1
    return mean * nu, (1 - mean) * nu


def sample_thresholds(mean, std, size):
    """Sample `size` happiness threshold values in [0, 1] from a Beta
    distribution with the given mean and standard deviation.

    If std <= 0, or mean is 0 or 1 (a Beta distribution can't have
    positive spread at those means), returns `mean` repeated instead.
    """
    if std <= 0 or mean <= 0 or mean >= 1:
        return np.full(size, mean)
    alpha, beta = beta_params(mean, std)
    return np.random.beta(alpha, beta, size=size)


def is_happy(like_neighbours, n_neighbours, threshold):
    """Return whether like_neighbours / n_neighbours meets threshold.

    This is the standard (unweighted) Schelling happiness rule.
    """
    return (float(like_neighbours) / n_neighbours) >= threshold


def is_happy_weighted(like_neighbour_mask, distances, threshold):
    """Return whether an inverse-distance-weighted proximity score meets threshold.

    ``like_neighbour_mask`` is a boolean array indicating which neighbours
    (given in the same order as ``distances``) belong to the same group as
    the agent. Only occupied neighbour cells are passed in (empty cells
    are excluded entirely, not counted as unlike), so an agent with no
    occupied neighbours is considered happy by default.

    The score is the fraction of total neighbourly "influence" (each
    neighbour weighted by 1/distance, so closer neighbours count more)
    that comes from same-group neighbours. This is always in [0, 1]: a
    neighbourhood that is 100% same-group scores exactly 1.0 regardless
    of distances, and a close out-of-group neighbour dilutes the score
    more than a distant one, same as close same-group neighbours boost
    it more than distant ones.
    """
    like_neighbour_mask = np.asarray(like_neighbour_mask, dtype=bool)
    distances = np.asarray(distances, dtype=float)
    if len(distances) == 0:
        return True
    weights = 1 / distances
    score = weights[like_neighbour_mask].sum() / weights.sum()
    return score >= threshold


class Agent:
    """Represent one agent in the Schelling segregation model."""

    def __init__(self, population, group, threshold, id_, is_happy=is_happy):

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
        self.neighbour_distances = np.array([], dtype=float)
        self.like_neighbour_mask = np.array([], dtype=bool)
        self.is_happy = is_happy

    def happy(self):
        """Return whether the agent is happy with its current neighbours."""
        if self.is_happy is is_happy_weighted:
            return self.is_happy(
                self.like_neighbour_mask,
                self.neighbour_distances,
                self.threshold,
            )
        return self.is_happy(
            self.like_neighbours, self.n_neighbours, self.threshold
        )

    def move(self, show=True):
        """Move the agent to a random empty location."""

        # TODO: Could make this more systematic?
        # Carry out looped search here maybe
        new_id = int(np.random.choice(self.population.empty_spaces))
        self.population.empty_spaces.append(self.id)
        self.population.group_at[self.id] = -1
        self.unshow()
        self.id = new_id
        self.population.empty_spaces.remove(self.id)
        self.population.group_at[self.id] = self.group

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
        threshold_stds=None,
        n_neighbours=12,
        cols=None,
        background_col=(0, 0, 0),
    ):

        self.display = dis
        self.n_agents = n
        self.probs = probs
        self.n_groups = len(probs)
        self.thresholds = thresholds
        self.threshold_stds = (
            np.zeros(self.n_groups) if threshold_stds is None
            else np.asarray(threshold_stds, dtype=float)
        )
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
        nearest_neighbours = getattr(layout, "nearest_neighbours", None)
        nearest_neighbour_distances = getattr(
            layout, "nearest_neighbour_distances", None
        )

        if centres_x is None or centres_y is None:
            raise ValueError(
                "Display object must provide real LED layout coordinates"
            )
        if nearest_neighbours is None or nearest_neighbour_distances is None:
            raise ValueError(
                "Display object must provide a nearest_neighbours table"
            )
        if n_neighbours > nearest_neighbours.shape[1]:
            raise ValueError(
                f"n_neighbours ({n_neighbours}) exceeds the number of "
                f"neighbours available in the nearest_neighbours table "
                f"({nearest_neighbours.shape[1]})"
            )

        coords_count = min(self.n_cells, len(centres_x), len(centres_y))
        self.coords_x = np.asarray(centres_x[:coords_count], dtype=float)
        self.coords_y = np.asarray(centres_y[:coords_count], dtype=float)
        self.nearest_neighbours = np.asarray(nearest_neighbours)
        self.nearest_neighbour_distances = np.asarray(
            nearest_neighbour_distances, dtype=float
        )

        # Maps led id -> group of the agent occupying it, or -1 if empty.
        # Kept up to date incrementally as agents move (see Agent.move()),
        # so neighbour occupancy can be looked up in O(1) per candidate.
        self.group_at = np.full(self.n_cells, -1, dtype=np.int32)

        self.empty_spaces = list(range(self.n_cells))

        groups = np.random.choice(self.n_groups, p=probs, size=n)
        agent_ids = np.random.choice(self.empty_spaces, size=n, replace=False)
        self.empty_spaces = list(
            set(self.empty_spaces) - {int(i) for i in agent_ids}
        )

        # Each agent's own threshold is sampled from a Beta distribution
        # centred on its group's threshold, rather than every agent in a
        # group sharing that exact value.
        agent_thresholds = np.empty(n, dtype=float)
        for g in range(self.n_groups):
            mask = groups == g
            count = int(mask.sum())
            if count:
                agent_thresholds[mask] = sample_thresholds(
                    thresholds[g], self.threshold_stds[g], count
                )

        self.agents = [
            Agent(
                self,
                group,
                agent_thresholds[i],
                agent_id,
                is_happy=is_happy_weighted,
            )
            for i, (group, agent_id) in enumerate(zip(groups, agent_ids))
        ]
        for agent in self.agents:
            self.group_at[agent.id] = agent.group

        self.last_agent = -1

    def update_agent_neighbours(self, agent):
        """Look up agent's occupied neighbours from the static
        nearest_neighbours table (filtered to the first n_neighbours
        candidate cells) and update its like-neighbour state."""

        candidates = self.nearest_neighbours[agent.id, : self.n_neighbours]
        occupied_mask = self.group_at[candidates] >= 0
        agent.neighbour_ids = candidates[occupied_mask]
        agent.neighbour_distances = self.nearest_neighbour_distances[
            agent.id, : self.n_neighbours
        ][occupied_mask]
        neighbour_groups = self.group_at[agent.neighbour_ids]
        agent.like_neighbour_mask = neighbour_groups == agent.group
        agent.like_neighbours = int(agent.like_neighbour_mask.sum())

    def update_all_agents(self):
        """Advance the population by one update round.

        Returns True if any agent moved during the round, otherwise False.
        """

        any_moved = False

        logger.info("Updating all agents...")
        for agent in self.agents:
            self.update_agent_neighbours(agent)

            if not agent.happy():
                searches = 0
                while not agent.happy():
                    agent.move(show=False)
                    any_moved = True
                    self.update_agent_neighbours(agent)
                    searches += 1
                    if searches > 50:
                        # logging.info("Gave up looking.")
                        break

                agent.show()
                logger.info("Agent %d moved.", agent.id)

        return any_moved

    def move_next_agent(self):
        """Find the next agent, after the last one moved, that wants
        to move, move it to a new location, and return it.

        Agents are checked in order starting after ``last_agent``,
        wrapping around the population. Returns None if no agent
        wants to move.
        """

        n = len(self.agents)

        for step in range(n):
            i = (self.last_agent + 1 + step) % n
            agent = self.agents[i]
            self.update_agent_neighbours(agent)

            if agent.happy():
                continue

            searches = 0
            while not agent.happy():
                agent.move(show=False)
                self.update_agent_neighbours(agent)
                searches += 1
                if searches > 50:
                    break

            agent.show()
            logger.info("Agent %d moved.", agent.id)
            self.last_agent = i
            return agent

        return None

    def show(self):
        """Show all agents on the LED array."""

        if self.agents:
            agent_ids = np.array(
                [agent.id for agent in self.agents], dtype=np.int32
            )
            agent_rgb = np.array(
                [agent.colour for agent in self.agents], dtype=np.uint8
            )
            self.display.set_leds(agent_ids, agent_rgb)

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


def main(n_neighbours=12):
    """Run the Schelling simulation loop until interrupted.

    n_neighbours: number of nearest neighbours used in each agent's
        happiness calculation. For the display's roughly hexagonal LED
        packing, 6 is the first shell of nearest neighbours and 18 is
        the first two shells (i.e. everything within twice the nearest-
        neighbour distance).
    """

    logger.info("\n\n------- Schelling Segregation Model Simulation -------\n")

    dis = Display1593()
    dis.connect()
    try:
        cols = list(COLOURS)

        while True:
            logger.info("Initializing population model...")

            # Randomly assign population and model parameters
            # Number of population groups
            # p = [0.5, 0.4, 0.1]
            # n_values = [2, 3, 4]
            # n_groups = np.random.choice(n_values, p=p)
            n_groups = 2

            # Happiness thresholds - each group's value is the mean of
            # that group's per-agent threshold distribution
            thresholds = np.random.choice(THRESHOLD_VALUES, size=n_groups)
            threshold_stds = np.full(n_groups, THRESHOLD_STD)

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
                threshold_stds=threshold_stds,
                n_neighbours=n_neighbours,
                cols=cols[0:n_groups],
            )

            logger.info("%d agents initialized.", n_agents)
            logger.info("%d population groups.", population.n_groups)
            logger.info("Distribution: %s", str(population.probs))
            logger.info("Thresholds: %s", str(thresholds.tolist()))
            logger.info("Threshold stds: %s", str(threshold_stds.tolist()))
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-neighbours",
        type=int,
        default=6,
        help="number of nearest neighbours used in the happiness "
        "calculation (e.g. 6 or 18; default: 6)",
    )
    args = parser.parse_args()
    main(n_neighbours=args.n_neighbours)
