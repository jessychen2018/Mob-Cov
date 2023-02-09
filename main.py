import agent_based_simulation
import numpy as np
from matplotlib import pyplot as plt
import copy


comb = [20, 40, 40, 40, 100]  # number of containers at each layer
c0, c1, d = 1.1, 0.35, 1  # parameters for human mobility
particlesbegin = 100
particlesend = 1000
TotalTime = 2000
initial_infected_number = 5
TestNum = 20
div = 30

n_particles = [int((particlesend-particlesbegin)/div*i+particlesbegin) for i in range(1, div+1)]
BreakPointTemp = []

for i in range(div):
    NumTemp = 0
    modeltemp = agent_based_simulation.ContainerModel(comb, c0, c1, d, n_particles[i], TotalTime,initial_infected_number)  
    for j in range(TestNum):
        model = copy.deepcopy(modeltemp)
        model.run_simulation()
        Num = model.InfectedNum
        NumTemp = NumTemp+Num
    NumTemp = NumTemp/TestNum
    J = NumTemp / n_particles[i] - np.array(range(1, TotalTime + 1)) / TotalTime
    BreakPoint = np.argwhere(J == max(J)).flatten()[0]
    BreakPointTemp.append(BreakPoint)

plt.figure(1)
plt.plot(n_particles, BreakPointTemp)
plt.show()
