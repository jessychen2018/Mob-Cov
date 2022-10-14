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
    return dict(zip(range(n_ticks), np.random.choice(range(0, n_ticks-1), n_ticks-1, replace=False)))


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


def visualization(pos_scales, comb, sign):
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
        # plt.show()
        plt.savefig('C:/Users/admin/Desktop/human mobility/'+str(ti)+'.png')
        plt.close()


def visualization_map(pos_scales, infect_pop):
    from matplotlib import pyplot as plt
    from matplotlib.patches import Ellipse
    import cv2

    Nt, Np, Nlevel = np.shape(pos_scales)
    l2_info = [[730, 240, 200], [1200, 350, 200], [405, 665, 130], [690, 695, 150], [1100, 830, 160], [1100, 1160, 150],
               [1470, 650, 130], [1600, 960, 130], [1220, 1600, 140], [1230, 1870, 170]]

    # plot t = 0
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    map = cv2.imread('map.jpg')[:, :, 0]
    map[map < 255] = 1.0
    map[map >= 255] = 0.0
    # level 1
    colors = ['peru', 'cornflowerblue', 'orange', 'seagreen', 'pink']
    ells = [Ellipse(xy=(955, 240), width=1100, height=600, angle=10),
            Ellipse(xy=(560, 687), width=700, height=500, angle=10),
            Ellipse(xy=(1100, 1000), width=500, height=900, angle=0),
            Ellipse(xy=(1550, 870), width=800, height=450, angle=70),
            Ellipse(xy=(1250, 1750), width=600, height=600, angle=0)]
    for id, e in enumerate(ells):
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_alpha(0.2)
        e.set_facecolor(colors[id])
    infect_pos, health_pos, all_pos = [], [], []
    for pii in range(Np):
        xc, yc, r_max = l2_info[pos_scales[0, pii, 1]]
        theta = np.random.uniform(0, 2 * np.pi)
        rp = np.random.uniform(0, r_max)
        px, py = int(xc + rp * np.cos(theta)), int(yc + rp * np.sin(theta))
        while ([px, py, 2] in health_pos + infect_pos):
            theta = np.random.uniform(0, 2 * np.pi)
            rp = np.random.uniform(0, r_max)
            px, py = int(xc + rp * np.cos(theta)), int(yc + rp * np.sin(theta))
        if pii in infect_pop[0]:  # infected
            infect_pos.append([px, py, 2])
        else:
            health_pos.append([px, py, 2])
        all_pos.append([px, py])

    infect_pos = np.asarray(infect_pos)
    health_pos = np.asarray(health_pos)
    plt.imshow(map, cmap='Greys', vmin=0, vmax=3)
    plt.scatter(health_pos[:, 0], health_pos[:, 1], s=health_pos[:, 2], c='k')
    plt.scatter(infect_pos[:, 0], infect_pos[:, 1], s=infect_pos[:, 2], c='r')
    plt.savefig('C:/Users/admin/Desktop/human mobility/figs/' + str(0) + '.png')
    plt.close()

    # plot t >= 1
    previous_pos = all_pos
    for ti in range(1, Nt):
        fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
        map = cv2.imread('map.jpg')[:, :, 0]
        map[map < 255] = 1.0
        map[map >= 255] = 0.0
        # level 1
        colors = ['rosybrown', 'cornflowerblue', 'orange', 'seagreen', 'pink']
        ells = [Ellipse(xy=(955, 240), width=1100, height=600, angle=10),
                Ellipse(xy=(560, 687), width=700, height=500, angle=10),
                Ellipse(xy=(1100, 1000), width=500, height=900, angle=0),
                Ellipse(xy=(1550, 870), width=800, height=450, angle=70),
                Ellipse(xy=(1250, 1750), width=600, height=600, angle=0)]
        for id, e in enumerate(ells):
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_alpha(0.2)
            e.set_facecolor(colors[id])
        # # level 2
        # ells = [Ellipse(xy=(730, 240), width=400, height=400, angle=0),
        #         Ellipse(xy=(1200, 350), width=400, height=400, angle=0),
        #         Ellipse(xy=(405, 665), width=250, height=400, angle=0),
        #         Ellipse(xy=(680, 695), width=250, height=400, angle=0),
        #         Ellipse(xy=(1100, 830), width=300, height=300, angle=0),
        #         Ellipse(xy=(1100, 1160), width=300, height=300, angle=0),
        #         Ellipse(xy=(1470, 650), width=400, height=300, angle=0),
        #         Ellipse(xy=(1600, 960), width=400, height=300, angle=0),
        #         Ellipse(xy=(1220, 1600), width=400, height=250, angle=0),
        #         Ellipse(xy=(1230, 1870), width=400, height=250, angle=0)]
        # for e in ells:
        #     ax.add_artist(e)
        #     e.set_clip_box(ax.bbox)
        #     e.set_facecolor('none')
        #     e.set_edgecolor('black')
        infect_pos, health_pos, all_pos = [], [], []
        for pii in range(Np):
            if sum(pos_scales[ti-1, pii, :]-pos_scales[ti, pii, :]==0)<3:  # if move
                xc, yc, r_max = l2_info[pos_scales[ti, pii, 1]]
                theta = np.random.uniform(0, 2 * np.pi)
                rp = np.random.uniform(0, r_max)
                px, py = int(xc + rp * np.cos(theta)), int(yc + rp * np.sin(theta))
                while ([px, py, 2] in health_pos+infect_pos):
                    theta = np.random.uniform(0, 2*np.pi)
                    rp = np.random.uniform(0, r_max)
                    px, py = int(xc+rp*np.cos(theta)), int(yc+rp*np.sin(theta))
            else:
                px, py = previous_pos[pii]

            if pii in infect_pop[ti]:  # infected
                infect_pos.append([px, py, 2])
            else:
                health_pos.append([px, py, 2])
            all_pos.append([px, py])

        infect_pos = np.asarray(infect_pos)
        health_pos = np.asarray(health_pos)

        import matplotlib.colors
        # cmap0 = matplotlib.colors.ListedColormap(['white', 'grey', 'black', 'red'])
        # boundaries = [0.0, 1.0, 2.0, 3.0]
        # norm0 = matplotlib.colors.BoundaryNorm(boundaries, cmap0.N, clip=True)
        plt.imshow(map, cmap='Greys', vmin=0, vmax=3)
        plt.scatter(health_pos[:,0], health_pos[:,1], s=health_pos[:, 2], c='k')
        plt.scatter(infect_pos[:,0], infect_pos[:,1], s=infect_pos[:,2], c='r')
        plt.savefig('C:/Users/admin/Desktop/human mobility/figs/' + str(ti) + '.png')
        plt.close()

        previous_pos = all_pos
    pdb.set_trace()

