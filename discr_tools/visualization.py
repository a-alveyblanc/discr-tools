import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm, colors


def plot_results(nodes, result, u=None, u_h=None, plot_solution=False):
    # plotting
    fig = plt.figure(layout='constrained')
    fig.suptitle("Error on $\Omega$");

    # 2d plot of error on domain
    ax2d = fig.add_subplot(1, 2, 1, aspect='equal')
    ax3d = fig.add_subplot(1, 2, 2, projection='3d')

    ax2d.set_xlabel("x")
    ax2d.set_ylabel("y")
    ax2d.scatter(nodes[0], nodes[1], c=result)

    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.scatter(nodes[0], nodes[1], result, c=result)

    # colorbar
    vmin = np.min(result)
    vmax = np.max(result)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    fig.colorbar(mappable=cm.ScalarMappable(norm=norm), location='bottom',
                 ax=[ax2d,ax3d])
    plt.show()

    # change to 1 to see solution
    if plot_solution:

        if u is None or u_h is None:
            raise TypeError("Must include solution to plot the solution")

        fig = plt.figure(layout='constrained')
        fig.suptitle(r"Solution on $\Omega$")

        ax2d = fig.add_subplot(1, 2, 1, aspect='equal')
        ax3d = fig.add_subplot(1, 2, 2, projection='3d')

        ax2d.scatter(nodes[0], nodes[1], c=u_h)
        ax3d.scatter(nodes[0], nodes[1], u_h, c=u_h)
        vmin = np.min(u_h)
        vmax = np.max(u_h)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        fig.colorbar(mappable=cm.ScalarMappable(norm=norm),
                     location='bottom', ax=[ax2d,ax3d])

        plt.show()
