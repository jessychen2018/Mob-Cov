import numpy as np
from collections import defaultdict
import functools
from functions import random_dictionary
from functions import extract_discrete_pl
import random

class ContainerModel:
    """container model (ref: the scales of human mobility)
       + cov19 transmission (ref: an agent-based model to evaluate the covid-19 transmission risks in facilities)"""
    def __init__(self, comb, c0, c1, d, npar, T, infectednum):
        self.comb = comb
        self.n_particles = npar
        self.total_time = T
        self.ActualTime = T
        self.InfectedSign = np.zeros((T,npar),dtype=np.int32)
        self.InfectedSign[0,0:infectednum] = 1
        self.InfectedNum=np.zeros(T,dtype=np.int32)
        self.AgentPri = np.random.uniform(low=0.1,high=0.3,size=npar)
        self.AgentPrc = np.random.uniform(low=0.2,high=0.4,size=npar)

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
                random_number = extract_discrete_pl(self.betas[n], self.comb[n]**2)
                scale_position = nested_dictionary[n][random_number]
                pos.append(scale_position)
            self.pos_scales[0, pi, :] = np.array(pos)
        return nested_dictionary

    def run_simulation(self):
        sign=0
        for ti in range(1, self.total_time):
            for pi in range(self.n_particles):
                if random.random()<self.AgentPrc[pi]:
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
                        new_cell = self.nested_dictionary[l][random_number]
                    my_position.append(new_cell)

                    # Select smaller containers
                    for n in range(l+1, len(self.p_change)):
                        random_number = extract_discrete_pl(self.betas[n], self.comb[n]**2)
                        new_cell = self.nested_dictionary[n][random_number]
                        my_position.append(new_cell)
                    self.pos_scales[ti, pi, :] = np.array(my_position)

                else:
                    self.pos_scales[ti, pi, :]=self.pos_scales[ti-1, pi, :]
            
            InfectedSignTemp = np.zeros(self.n_particles,dtype=np.int32)    
            for i in range(self.n_particles):
                if self.InfectedSign[ti - 1, i]==1:
                    InfectedSignTemp[i] = 1
                    for j in range(self.n_particles):                        
                        if self.InfectedSign[ti-1,j]==0 and list(self.pos_scales[ti,i,:len(self.comb)-1])==list(self.pos_scales[ti,j,:len(self.comb)-1]) and random.random() < self.AgentPri[j]:
                            InfectedSignTemp[j] = 1
            self.InfectedSign[ti,:] = InfectedSignTemp

            for i in range(self.n_particles):
                if self.InfectedSign[ti,i]==True:
                    self.InfectedNum[ti]+=1
            if self.InfectedNum[ti]/self.n_particles>0.9 and sign == 0:
                sign=1
                self.ActualTime=ti+100
            if ti==self.ActualTime:
                self.InfectedNum[self.ActualTime:]=self.InfectedNum[self.ActualTime-1]
                break
        

    def BreakPoint(self):
        J = self.InfectedNum[0:self.ActualTime] / self.n_particles - np.array(range(1, self.ActualTime + 1)) / self.ActualTime
        BreakPoint = np.argwhere(J == max(J)).flatten()[0]
        return BreakPoint
