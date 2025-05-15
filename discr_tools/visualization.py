from discr_tools.discretization import Discretization
from discr_tools.nodal import EquispacedNodalBasis

import matplotlib.pyplot as plt
import numpy as np



def plot_contourf(u, discr):
    order = discr.order
    a, b = discr.interval
    dim = discr.dim
    nelts_1d = discr.nelts_1d

    fig = plt.figure(layout="constrained")

    solution_ax = fig.add_subplot()

    solution_ax.set_title("Numerical solution contour")

    nelts, _ = u.shape
    npts_1d = order+1
    u = u.reshape(nelts, *(npts_1d,)*2, order="F")

    vis_discr = Discretization(order, a, b, dim, nelts_1d,
                               basis_cls=EquispacedNodalBasis)

    ref_nodes = vis_discr.basis_cls.nodes
    vdm_vis = np.array([
        phi(ref_nodes)
        for phi in discr.basis_cls.basis
    ]).T

    u_vis = np.zeros((nelts, ref_nodes.shape[0]**dim)).reshape(
        nelts, *(ref_nodes.shape[0],)*dim, order="F"
    )
    v = np.einsum("il,elj->eij", vdm_vis, u)
    u_vis = np.einsum("jl,eil->eij", vdm_vis, v)
    u = u_vis
    npts_1d = ref_nodes.shape[0]

    x, y = vis_discr.mapped_elements
    x = x.reshape(nelts, *(npts_1d,)*2, order="F")
    y = y.reshape(nelts, *(npts_1d,)*2, order="F")

    vmin = np.min(u)
    vmax = np.max(u)

    for ielt in range(nelts):
        solution_ax.pcolormesh(x[ielt], y[ielt],
                               u[ielt],
                               vmin=vmin,
                               vmax=vmax, shading='gouraud')

    plt.savefig("poisson-visualization.pdf")
