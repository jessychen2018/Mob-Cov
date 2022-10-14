import agent_based_simulation
import random
import numpy as np
from matplotlib import pyplot as plt
import pdb
import time
import copy
import xlwt

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
n_particles = [int((particlesend-particlesbegin)/30*i+particlesbegin) for i in range(1,div+1)]

for m in range(div):    
    modeltemp = agent_based_simulation.ContainerModel(comb, c0, c1, d, n_particles[m], TotalTime,initial_infected_number)
    workbook = xlwt.Workbook(encoding= 'ascii')
    for i in range(TestNum):
        model=copy.deepcopy(modeltemp)
        model.run_simulation()
        worksheet = workbook.add_sheet("Sheet"+str(i+1))
        for j in range(model.n_particles):
            worksheet.col(j).width = 256*30
            for k in range(model.total_time):
                worksheet.write(k,j, str(model.pos_scales[k,j,:])+" "+str(model.InfectedSign[k,j]))
    workbook.save("PosAndSignData"+str(m+1)+".xls")

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
