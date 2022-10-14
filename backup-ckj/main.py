import agent_based_simulation
from functions import visualization
import random
import numpy as np
from matplotlib import pyplot as plt
import pdb
import time
import copy

comb = [5, 10, 10, 10, 30]
c0, c1, d = 1.1, 0.35, 1.3
n_particles = 1000
TotalTime = 2000
initial_infected_number = 5

# probability distribution of step size at the last layer
# model = agent_based_simulation.ContainerModel(comb, c0, c1, d, n_particles, TotalTime, initial_infected_number)
# model.run_simulation()
# def cal_value_dist(x, xmin, xmax, num_bins):
#     """ x is a list
#         calculate prob of values in x """
#     import seaborn as sns
#     values = [[h.xy[0], h.get_width(), h.get_height()] for h in
#               sns.distplot(x, kde=False, bins=np.logspace(np.log10(xmin), np.log10(xmax), num_bins), hist_kws={'density': True}).patches]
#     pdb.set_trace()
#     xx, p_xx = [], []
#     for i, value in enumerate(values):
#         xx.append(value[0] + value[1] / 2)
#         p_xx.append(value[2])
#     plt.close()
#     return xx, p_xx/np.sum(p_xx)
# lp_all, p_lp = cal_value_dist(model.record, xmin=10, xmax=500, num_bins=40)
# fig, ax = plt.subplots(figsize=(6, 4))
# plt.scatter(lp_all, p_lp, color='black', s=20)
# plt.ylim(1e-3, 3e-1)
# plt.xlim(0.9e1, 1.1e3)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.set_yticks([1e-2, 1e-1])
# ax.set_yticklabels([1e-2, 1e-1], fontsize=20)
# ax.set_xticks([1e1, 1e2, 1e3])
# ax.set_xticklabels([1e1, 1e2, 1e3], fontsize=20)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.yscale('log')
# plt.xscale('log')
# ax.set_xlabel('Step size (s)', fontsize=22)
# ax.set_ylabel('Porbability', fontsize=22)
# plt.tight_layout()
# plt.show()
# pdb.set_trace()

# probability of choose a layer l, influenced by d
model = agent_based_simulation.ContainerModel(comb, c0, c1, d, n_particles, TotalTime, initial_infected_number)
model.run_simulation()
from collections import Counter
counts = Counter(model.record)
x = counts.keys()
y = counts.values()
y = [i*1.0/sum(y) for i in y]
y = sorted(y, reverse=True)

xx = np.arange(1.0, 6.0, 1.0)
yy = np.exp(-d*(xx-1))
yy = yy/sum(yy)

fig, ax = plt.subplots(figsize=(6, 4))
plt.bar([1, 2, 3, 4, 5], y, color='grey', width=0.6)
plt.plot(xx, yy, color='black', linestyle='dashed', linewidth=2)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.ylim(0, 0.6)
ax.set_yticks([0.0, 0.2, 0.4, 0.6])
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_xticklabels([1, 2, 3, 4, 5], fontsize=20, color='black')
ax.set_yticklabels([0.0, 0.2, 0.4, 0.6], fontsize=20, color='black')
ax.set_ylabel('Probability', fontsize=22)
ax.set_xlabel('Container layer', fontsize=22)
plt.tight_layout()
plt.show()
pdb.set_trace()

