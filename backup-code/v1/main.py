from utils import utils2
from utils import scale_by_scale_optim
from utils import agent_based_simulation_on_grid2
import random
import numpy as np
import time
from matplotlib import pyplot as plt
import pdb

### Generate parameters
L = 3
comb = [random.randint(20,100) for i in range(L)]
c0 = random.random()*2
c1 = random.random()*2
d = random.random()*2

### Run simulation
model = agent_based_simulation_on_grid2.ContainerModel(comb, c0, c1, d)
model.run_simulation(1)
pdb.set_trace()
x,y= zip(*model.positions)

### Visualize
#Colors of transitions
transitions = utils2.get_scale_labels(model.positions_scales)

plt.scatter(x,y,color = 'black',s=1)

# xs = [[i,i2] for (i,i2) in zip(x[:-1],x[1:])]
# ys = [[j,j2] for (j,j2) in zip(y[:-1],y[1:])]
#
# plt.plot(xs,ys, alpha=0.5)
plt.show()