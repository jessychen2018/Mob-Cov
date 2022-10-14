import agent_based_simulation
from functions import visualization_map

import random
import numpy as np
from matplotlib import pyplot as plt
import time
import copy
import pdb

### parameters
comb = [20,40,40,40,100]#[5, 11, 800]#[random.randint(20, 100) for i in range(Levels)]  # number of containers
c0, c1, d = 1.1,0.35,1#0.5, 0.15, 3.0#random.random()*2, random.random()*2, random.random()*2
n_particles = 1000 # number of particles
TotalTime = 60

J_all = []
for iid in range(200):
    model = agent_based_simulation.ContainerModel(comb=comb, c0=c0, c1=c1, d=d, npar=n_particles, T=TotalTime,
                                                  Pri=[0.1,0.3], Prc=[0.2,0.4], initial_infected=[0,1,2,3,4])
    model.run_simulation()
    J, maxJ, max_iter = model.cost_functon()
    J_all.append(J)
    print(str(iid))
plt.plot(np.arange(TotalTime), np.mean(np.array(J_all), axis=0))
plt.show()
# visualization_map(model.pos_scales, model.InfectedPop)
pdb.set_trace()


