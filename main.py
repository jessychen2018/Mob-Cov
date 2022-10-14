import agent_based_simulation
import random
import numpy as np
from matplotlib import pyplot as plt
import pdb
import time
import copy

start=time.perf_counter()
#500 100 10 
### Generate parameters
comb = [20,40,40,40,100]#[random.randint(20, 100) for i in range(Levels)]  # number of containers
c0, c1, d = 1.1,0.35,1#random.random()*2, random.random()*2, random.random()*2
particlesbegin= 100
particlesend= 1000  # number of particles
TotalTime = 2000
initial_infected_number = 5
TestNum=20
div=30

n_particles = [int((particlesend-particlesbegin)/div*i+particlesbegin) for i in range(1,div+1)]
BreakPointTemp=[]

for i in range(div):
    print(str(i))
    NumTemp=0
    modeltemp = agent_based_simulation.ContainerModel(comb, c0, c1, d, n_particles[i], TotalTime,initial_infected_number)  
    for j in range(TestNum):
        model=copy.deepcopy(modeltemp)
        model.run_simulation()
        Num = model.InfectedNum
        NumTemp=NumTemp+Num
    NumTemp=NumTemp/TestNum
    J = NumTemp / n_particles[i] - np.array(range(1, TotalTime + 1)) / TotalTime
    BreakPoint = np.argwhere(J == max(J)).flatten()[0]
    BreakPointTemp.append(BreakPoint)

plt.figure(1)
plt.plot(n_particles, BreakPointTemp)
plt.show()

"""
plt.figure(1)
plt.plot(range(1, TotalTime + 1), NumTemp)
plt.scatter(BreakPoint,NumTemp[BreakPoint], color="red")

plt.figure(2)
plt.plot(range(1, 1 + TotalTime), J)
plt.scatter(BreakPoint, J[BreakPoint], color="red")
plt.show()
"""

"""
from functions import random_dictionary, visualization
visualization(model.pos_scales, model.comb, model.InfectedSign, model.InfectedNum, model.total_time)
"""

end =time.perf_counter()
print("运行耗时" , end-start)