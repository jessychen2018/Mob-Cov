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


def scales_to_loc(scales_vector, comb, space=True):
    ticks_prod = [1] + list(np.cumprod(comb))
    size_of_the_box = ticks_prod[-1]

    x = int(sum([np.floor((size_of_the_box/ticks_prod[1 + s])**0.5) * (c%comb[s]) for (s, c) in
                 enumerate(scales_vector)]))
    y = int(sum([np.floor((size_of_the_box/ticks_prod[1 + s])**0.5) * np.floor(c/comb[s]) for (s, c) in
                 enumerate(scales_vector)]))
    # Add space
    if space:
        space_x = sum([100 * int(x / k**0.2) for (k) in ticks_prod]) - x
        space_y = sum([100 * int(y / k**0.2) for (k) in ticks_prod]) - y
        x += space_x
        y += space_y
    return x, y


def visualization(pos_scales, comb, sign, infectednum, totaltime):
    from matplotlib import pyplot as plt
    nt, npar, n_scales = np.shape(pos_scales)
    pos = np.zeros((nt, npar, 2))
    for ti in range(nt):
        AgentPosTemp1x=[]
        AgentPosTemp1y=[]
        for pi in range(npar):
            if sign[ti,pi]==1:
                AgentPosTemp1x.append(scales_to_loc(pos_scales[ti, pi, :], comb)[0])
                AgentPosTemp1y.append(scales_to_loc(pos_scales[ti, pi, :], comb)[1])
            pos[ti, pi, :] = scales_to_loc(pos_scales[ti, pi, :], comb)
        plt.scatter(pos[ti, :, 0], pos[ti, :, 1], c=pos_scales[ti, :, 0], s=3)
        plt.scatter(AgentPosTemp1x, AgentPosTemp1y, c="r", s=3)
        plt.savefig('C:/Users/LiYanQing/Desktop/human mobility/figs1/'+str(ti)+'.png')
        plt.close()

    """
    plt.plot(range(1,totaltime+1),infectednum)
    plt.savefig('C:/Users/LiYanQing//Desktop/human mobility/figs1/'+str(totaltime)+'.png')
    plt.close()

    J = infectednum / npar - np.array(range(1, totaltime + 1)) / totaltime
    BreakPoint = np.argwhere(J == max(J)).flatten()[0]
    plt.plot(range(1, 1 + totaltime), infectednum)
    plt.scatter(BreakPoint,infectednum[BreakPoint], color="red")
    plt.savefig('C:/Users/LiYanQing//Desktop/human mobility/figs1/'+str(totaltime+1)+'.png')
    plt.close()
    plt.plot(range(1, 1 + totaltime), J)
    plt.scatter(BreakPoint, J[BreakPoint], color="red")
    plt.savefig('C:/Users/LiYanQing//Desktop/human mobility/figs1/' + str(totaltime + 2) + '.png')
    plt.close()
    # plt.pause(0.1)
    # pdb.set_trace()
    """

