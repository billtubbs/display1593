#!/usr/bin/python
#
#   Implementation of the Schelling Segregation Model
#
#   As described here:
#    http://quant-econ.net/py/schelling.html
#
#   Using:
#    - John Zelle's Graphics Module, and
#    - the KDTree class from scipy.spatial
#


import numpy as np
from scipy.spatial import KDTree
from itertools import izip
from datetime import datetime
import time
from random import shuffle

import display1593 as display


class Agent(object):
    """Agent class for simulating an agent in a Schelling
    segregation model.
    """

    def __init__(self, population, group, threshold):

        self.population = population
        self.group = group
        self.threshold = threshold
        self.colour = self.population.colours[group]
        self.id = np.random.choice(self.population.empty_spaces)
        self.population.empty_spaces.remove(self.id)
        self.location = (
            display.leds.centres_x[self.id],
            display.leds.centres_y[self.id]
        )
        self.n_neighbours = self.population.n_neighbours

    def happy(self):
        """Calculates agent's happiness based on current
        neighbours.
        """

        h = (float(self.like_neighbours) / self.n_neighbours) >= self.threshold
        #print "Agent's happiness: (%d/%d) >= %4.1f = %s" % \
        #   ((self.like_neighbours, self.n_neighbours, self.threshold, str(h)))

        return h

    def move(self, show=True):
        """Moves agent to a random new location.
        """

        # TODO: Could make this more systematic?
        # Carry out looped search here maybe
        self.new_id = np.random.choice(self.population.empty_spaces)

        self.population.empty_spaces.append(self.id)
        self.unshow()
        self.id = self.new_id
        self.population.empty_spaces.remove(self.id)

        self.location = (
            display.leds.centres_x[self.id],
            display.leds.centres_y[self.id]
        )
        if show:
            self.show()

    def show(self):
        """Show the agent on the LED array by lighting the
        appropriate LED with the agent's group colour.
        """

        self.population.display.setLed(self.id, self.colour)

    def unshow(self):
        """Clear the LED representing the agent.
        """

        self.population.display.setLed(self.id, self.population.background_col)


class Population(object):
    """Population class for simulating a population of
    agents in a Schelling segregation model.
    """

    def __init__(self, dis, n, probs, thresholds, n_neighbours=9, cols=None,
                 background_col=(0, 0, 0)):

        self.display = dis
        self.n_agents = n
        self.n_groups = len(probs)
        self.probs = probs
        if cols == None:
            colour_set = display.leds.colourArray8[1:(self.n_groups + 1)]
            cols = [(col >> 16, (col >> 8) % 256, col % 256) for col in
                    colour_set]
        self.colours = cols
        self.background_col = background_col
        self.agents = []
        self.n_neighbours = n_neighbours

        self.empty_spaces = range(display.leds.numCells)

        self.agents = [Agent(self, group, thresholds[group]) for group in
            np.random.choice(self.n_groups, p=probs, size=n)]

    def count_like_neighbours(self, agent):

        assert len(agent.neighbour_ids) == self.n_neighbours
        tally = dict.fromkeys(range(self.n_groups), 0)

        for n in agent.neighbour_ids:
            n_grp = self.agents[n].group
            tally[n_grp] = tally.get(n_grp, 0) + 1

        agent.like_neighbours = tally[agent.group]

    def update_agents(self):
        """Updates the locations of all agents in the model
        once. Returns False if the mouse was clicked in the
        window, otherwise True.
        """

        # build a list of all agent locations
        # This would be faster if they were already in one array
        all_locations = [agent.location for agent in self.agents]

        # Flag used to detect when no agents moved in one round
        any_moved = False
        moved = True

        print "Updating all agents..."
        for i, agent in enumerate(self.agents):

            #print "Checking neighbours for agent", i, "group:", agent.group

            # Build a KDTree from all agent locations
            if moved:
                tree = KDTree(all_locations)
                #print "KD-Tree rebuilt"
            moved = False

            # Query the KDTree to find the k nearest neighbours.
            # KDTree.query returns two arrays, the first contains the
            # nearest neighbour distances, the second contains the
            # indeces of the nearest neighbours. Here, we ignore the
            # first row as this is the location of the current agent.
            k = self.n_neighbours + 1
            agent.neighbour_ids = tree.query(agent.location, k=k)[1][1:]

            #print "Agent's neighbours:", agent.neighbour_ids, \
            #    [self.agents[a].group for a in agent.neighbour_ids]
            self.count_like_neighbours(agent)
            #print "Agent has %d like neighbours." % agent.like_neighbours

            if not agent.happy():

                #print "Agent not happy..."
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

                    #print "Agent's neighbours:", agent.neighbour_ids, \
                    #    [self.agents[a].group for a in agent.neighbour_ids]

                    self.count_like_neighbours(agent)

                    #print "Agent has", agent.like_neighbours, \
                    #       "like neighbours."
                    searches += 1
                    if searches > 50:
                        #print "Gave up looking."
                        break

                # Put the current agent's location back in the list
                all_locations.insert(i, agent.location)

            t = datetime.now()

            if moved == True:
                agent.show()
                print "%02d:%02d:%02d Agent %d moved." % (t.hour, t.minute,
                                                          t.second, i)
            #raw_input("Press enter to contine...")

        return any_moved

    def show(self):
        """Show all agents on the LED array.
        """

        for agent in self.agents:
            agent.show()

        for i in self.empty_spaces:
            self.display.setLed(i, self.background_col)

    def unshow(self):
        """Clear all agents on the LED array."""

        for agent in self.agents:
            agent.unshow()


def main():

    print "\n------- Schelling Segregation Model Simulation -------\n"

    # Get current time
    start_time = datetime.now()
    hr, mn, sc = (start_time.hour, start_time.minute, start_time.second)
    print "Date: %02d/%02d/%4d" % (start_time.day, start_time.month, start_time.year)

    # Connect to LED display
    dis = display.Display1593()
    dis.connect()

    cols = [
        display.leds.colour['orange'],
        display.leds.colour['green'],
        display.leds.colour['grey'],
        display.leds.colour['blue'],
        display.leds.colour['brown'],
        display.leds.colour['yellow'],
        display.leds.colour['dark red']
    ]

    while True:
        dis.clear()

        print "Initializing population model..."
        # Define population and model parameters
        n_agents = display.leds.numCells - 400

        # Randomly sort the colours
        shuffle(cols)

        p = [0.5, 0.4, 0.1]
        n_groups = np.random.choice(range(2, 5), p=p)
        x = [(np.random.rand() + 0.25) for i in range(n_groups)]
        t = sum(x)
        probs = [p/t for p in x]

        thresholds = np.random.choice([0.25, 0.35, 0.5], size=n_groups)
        n_neighbours = 9

        population = Population(dis, n_agents, probs, thresholds,
                                n_neighbours=n_neighbours,
                                cols=cols[0:n_groups])

        print n_agents, " agents initialized."
        print population.n_groups, "groups"
        print "Distribution:", population.probs
        print "Thresholds:", thresholds
        print "Number of nearest neighbours:", n_neighbours

        print "Displaying initial population..."
        population.show()

        print "Model updating started..."

        while population.update_agents():
            pass

        print "Stable population reached"
        time.sleep(120)

    print "Results"
    print "   #:   id,  g,       x,       y, nn, neighbour_ids"
    for i, agent in enumerate(population.agents):
        print "%4d: %4d, %2d, %7.2f, %7.2f, %2d," % \
            (
                i,
                agent.id,
                agent.group,
                agent.location[0],
                agent.location[1],
                agent.like_neighbours
            ), \
            agent.neighbour_ids

if __name__ == "__main__":
    main()

