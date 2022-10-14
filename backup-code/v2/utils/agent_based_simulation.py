import numpy as np
from collections import defaultdict
import functools
from utils.functions import random_dictionary
from utils.functions import extract_discrete_pl
import pdb


class ContainerModel:
    """container model (ref: the scales of human mobility)
       + cov19 transmission (ref: an agent-based model to evaluate the covid-19 transmission risks in facilities)"""
    def __init__(self, comb, c0, c1, d, npar, T, n_infected):
        self.comb = comb
        self.n_particles = npar
        self.total_time = T
        self.infected = [list(np.random.choice(a=range(npar), size=n_infected))]

        # parameters of the bernullis and power-laws distributions
        self.p_change = [np.exp(-d*s) for s in reversed(range(1, len(self.comb)+1))]
        self.p_change = self.p_change / sum(self.p_change)
        self.betas = [c0+c1*i for i in reversed(range(len(self.comb)))]

        self.pos_scales = np.zeros((self.total_time, self.n_particles, len(self.comb)), dtype=np.int)
        self.nested_dictionary = self.initialize_simulation()

    def initialize_simulation(self):
        nested_dictionary = []
        for n in range(len(self.comb)):
            aa = defaultdict(functools.partial(random_dictionary, self.comb[n]))
            nested_dictionary.append(aa)

        for pi in range(self.n_particles):
            pos = []
            for n in range(len(self.comb)):
                random_number = extract_discrete_pl(self.betas[n], self.comb[n]**2)
                scale_position = nested_dictionary[n][tuple(pos)][random_number]
                pos.append(scale_position)
            self.pos_scales[0, pi, :] = np.array(pos)
        return nested_dictionary

    def run_simulation(self):
        for ti in range(1, self.total_time):
            for pi in range(self.n_particles):
                my_position = []

                # CHOSE l*
                l = np.random.choice(a=range(len(self.p_change)), size=1, p=self.p_change)[0]

                # Copy larger containers
                for li in range(0, l):
                    my_position.append(self.pos_scales[ti-1, pi, li])

                # select new container (!= from current)
                old_cell = self.pos_scales[ti-1, pi, l]
                new_cell = old_cell
                while (new_cell == old_cell):
                    random_number = extract_discrete_pl(self.betas[l], self.comb[l]**2)
                    new_cell = self.nested_dictionary[l][tuple(my_position)][random_number]
                my_position.append(new_cell)

                # Select smaller containers
                for n in range(l+1, len(self.p_change)):
                    random_number = extract_discrete_pl(self.betas[n], self.comb[n]**2)
                    new_cell = self.nested_dictionary[n][tuple(my_position)][random_number]
                    my_position.append(new_cell)

                self.pos_scales[ti, pi, :] = np.array(my_position)


