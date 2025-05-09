import numpy as np

import matplotlib.pyplot as plt


def plot_solution(discr, u, fig_name=None):
    """
    Expects *u* to have shape (nelts, ndofs). Only supports 2D plots
    """

    dim, nelts, _ = discr.mapped_elements.shape
    npts_1d = discr.order + 1

    x = discr.mapped_elements.reshape(dim, nelts, *(npts_1d,)*dim, order="F")
    u = u.reshape(nelts, *(npts_1d,)*dim, order="F")

    ax = plt.axes(projection='3d')
    for ielt in range(nelts):
        ax.plot_surface(
            x[0, ielt],
            x[1, ielt],
            u[ielt],
            vmin=np.min(u),
            vmax=np.max(u),
            cmap='viridis'
        )

    if fig_name is None:
        fig_name = "solution.png"

    plt.savefig(fig_name)
