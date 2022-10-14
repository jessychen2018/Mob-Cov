import pdb

from matplotlib import container
import numpy as np

def random_dictionary(n_ticks):
    '''Return a dictionary where keys are the range from 0 to n_ticks and values are the shuffled values.

    Input
    -----
    n_ticks: (int)
        Number of containers

    Output
    ------
    (dict):
        For each key it assign a shuffled label.

    '''
    return dict(zip(range(n_ticks**2), np.random.choice(range(0, n_ticks**2-1), n_ticks**2-1, replace=False)))


def extract_discrete_pl(beta, xmax, xmin=1):
    '''Return an integer number between xmin-1 and xmax extracted from a power law distribution P(x)~x^(-beta).

    Input
    -----
    xmin: (float) (>=1)
        Minimum value.
    xmax: (float)
        Maximum value.
    beta: (float)
        Exponent.

    Output
    ------
    (float): value from the power law.

    '''

    u = np.random.rand()
    dt2 = ((u)*(xmax**(1-beta)-xmin**(1-beta)) + xmin**(1-beta))**(1./(1-beta))

    return int(dt2) - 1


def scales_to_loc(scales_vector):
    xx = [[100, 300, 250, 70], [550, 600, 500, 400], [780, 800, 850, 910], [800, 700, 680, 710],
         [870, 830, 950, 780], [600, 500, 550, 400], [600, 630, 400, 410], [310, 80, 250, 100]]
    yy = [[50, 100, 300, 310], [30, 200, 310, 380], [50, 100, 300, 200], [450, 600, 400, 680],
         [750, 800, 800, 850], [730, 800, 900, 600], [500, 650, 600, 500], [700, 750, 850, 900]]
    x = xx[scales_vector[0]][scales_vector[1]] + np.random.uniform(-0.3, 0.3)*scales_vector[2] + np.random.uniform(-0.1, 0.1)*scales_vector[3]
    y = yy[scales_vector[0]][scales_vector[1]] + np.random.uniform(-0.3, 0.3)*scales_vector[2] + np.random.uniform(-0.1, 0.1)*scales_vector[3]
    return x, y


def visualization(pos_scales, comb, sign):
    from matplotlib import pyplot as plt
    colors = ['grey', 'black', 'orange', 'gold', 'lightcoral', 'lightsteelblue', 'royalblue', 'wheat', 'lightgrey']
    nt, npar, n_scales = np.shape(pos_scales)

    for ti in range(20):
        pos_x = [[], [], [], [], [], [], [], [], []]
        pos_y = [[], [], [], [], [], [], [], [], []]
        infect_x, infect_y = [], []
        for pi in range(npar):
            if sign[ti, pi] == 1:
                temp_x, temp_y = scales_to_loc(pos_scales[ti, pi, :])
                infect_x.append(temp_x)
                infect_y.append(temp_y)
            temp_x, temp_y = scales_to_loc(pos_scales[ti, pi, :])
            pos_x[pos_scales[ti, pi, 0]].append(temp_x)
            pos_y[pos_scales[ti, pi, 0]].append(temp_y)

        fig, ax = plt.subplots(figsize=(6, 4))
        for gi in range(9):
            plt.scatter(pos_x[gi], pos_y[gi], c=colors[gi], s=8)
        plt.scatter(infect_x, infect_y, c="red", s=10)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xticklabels([], fontsize=14, color='black')
        ax.set_yticklabels([], fontsize=14, color='black')

        ax.set_xlabel('', fontsize=18)
        ax.set_ylabel('', fontsize=18)
        ax.legend([], [], frameon=False)
        plt.tight_layout()
        # plt.show()
        # pdb.set_trace()
        # plt.show()
        # pdb.set_trace()
        plt.savefig('D:/Current Research/human mobility/backup-ckj/figs/'+str(ti)+'.png')
        plt.close()

