from utils import agent_based_simulation
import random
import numpy as np
from matplotlib import pyplot as plt
import pdb

### Generate parameters
Levels = 5
comb = [random.randint(20, 100) for i in range(Levels)]  # number of containers
c0, c1, d = random.random()*2, random.random()*2, random.random()*2
n_particles = 500  # number of particles
TotalTime = 100
initial_infected_number = 5

### Run simulation
model = agent_based_simulation.ContainerModel(comb, c0, c1, d, n_particles, TotalTime, initial_infected_number)
model.run_simulation()

from utils.functions import visualization
visualization(model.pos_scales, model.comb)
