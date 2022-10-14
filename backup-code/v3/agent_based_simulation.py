import pdb

import numpy as np
from collections import defaultdict
import functools
from functions import random_dictionary
from functions import extract_discrete_pl
import random


class ContainerModel:
    """container model (ref: the scales of human mobility)
       + cov19 transmission (ref: an agent-based model to evaluate the covid-19 transmission risks in facilities)"""
    def __init__(self, comb, c0, c1, d, npar, T, Pri, Prc, initial_infected):
        self.comb = comb
        self.n_particles = npar
        self.total_time = T
        self.InfectedPop = [initial_infected]  # store id
        self.AgentPri = np.random.uniform(low=Pri[0], high=Pri[1], size=npar)
        self.AgentPrc = np.random.uniform(low=Prc[0], high=Prc[1], size=npar)

        self.p_change = [np.exp(-d*s) for s in reversed(range(1, len(self.comb)+1))]
        self.p_change = self.p_change / sum(self.p_change)
        self.betas = [c0+c1*i for i in reversed(range(len(self.comb)))]

        self.pos_scales = np.zeros((self.total_time, self.n_particles, len(self.comb)), dtype=np.int)
        self.nested_dictionary = self.initialize_simulation()

    def initialize_simulation(self):
        nested_dictionary = []
        for n in range(len(self.comb)):
            aa = random_dictionary(self.comb[n])
            nested_dictionary.append(aa)

        for pi in range(self.n_particles):
            pos = []
            for n in range(len(self.comb)):
                random_number = extract_discrete_pl(self.betas[n], self.comb[n])
                scale_position = nested_dictionary[n][random_number]
                pos.append(scale_position)
            self.pos_scales[0, pi, :] = np.array(pos)
        return nested_dictionary

    def run_simulation(self):
        for ti in range(1, self.total_time):
            # container model for particle movements
            for pi in range(self.n_particles):
                if random.random()<self.AgentPrc[pi]:  # check if moving
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
                        random_number = extract_discrete_pl(self.betas[l], self.comb[l])
                        new_cell = self.nested_dictionary[l][random_number]
                    my_position.append(new_cell)

                    # Select smaller containers
                    for n in range(l+1, len(self.p_change)):
                        random_number = extract_discrete_pl(self.betas[n], self.comb[n])
                        new_cell = self.nested_dictionary[n][random_number]
                        my_position.append(new_cell)
                    self.pos_scales[ti, pi, :] = np.array(my_position)

                else:
                    self.pos_scales[ti, pi, :]=self.pos_scales[ti-1, pi, :]

            # Infection
            temp, potential_ids = [], []
            for infected_i in self.InfectedPop[ti-1]:
                check_temp = np.sum(self.pos_scales[ti,:,:-1] == self.pos_scales[ti, infected_i, :-1], axis=1)
                potential_ids += np.where(check_temp==len(self.comb)-1)[0].tolist()
            potential_ids = list(set(potential_ids))
            for pid in potential_ids:
                if random.random()<self.AgentPri[pid]:  # get infected
                    temp.append(pid)
            temp += self.InfectedPop[ti-1].copy()
            temp = list(set(temp))
            self.InfectedPop.append(temp)

    def cost_functon(self):
        J = np.zeros(self.total_time, float)
        for ti in range(self.total_time):
            J[ti] = len(self.InfectedPop[ti])*1.0/self.n_particles - ti*1.0/self.total_time
        return J, np.max(J), np.where(J==np.max(J))




