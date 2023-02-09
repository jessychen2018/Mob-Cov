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


