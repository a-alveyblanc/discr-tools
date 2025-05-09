import discr_tools.geometry as geo
import discr_tools.kernels as knl

from functools import partial

import numpy as np


def poisson_matvec(queue, discr):
    dim, nelts, npts = discr.mapped_elements.shape
    npts_1d = discr.order + 1

    inv_jac_t = geo.inverse_jacobian_t(discr.mapped_elements, discr.basis_cls)
    det_j = geo.jacobian_determinant(discr.mapped_elements, discr.basis_cls)

    inv_j_t_tp = inv_jac_t.reshape(dim, dim, nelts, *(npts_1d,)*dim, order="F")
    det_j_tp = det_j.reshape(nelts, *(npts_1d,)*dim, order="F")

    wts = discr.basis_cls.weights

    if dim == 2:
        spec = "kiexy,kjexy,exy,x,y->ijexy"
    elif dim == 3:
        spec = "kiexyz,kjexyz,exyz,x,y,z->ijexyz"
    else:
        raise ValueError("Only supported for dim = 2, 3")

    g = np.einsum(spec, inv_j_t_tp, inv_j_t_tp, det_j_tp, *(wts,)*dim)

    def matvec(u):
        u_g = discr.gather(u).reshape(nelts, *(npts_1d,)*dim, order="F")

        d = discr.operators.diff_operator

        if dim == 2:
            ur = np.einsum("il,elj->eij", d, u_g)
            us = np.einsum("jl,eil->eij", d, u_g)

            ux = g[0,0]*ur + g[0,1]*us
            uy = g[1,0]*ur + g[1,1]*us

            uxx = np.einsum("li,elj->eij", d, ux)
            uyy = np.einsum("lj,eil->eij", d, uy)

            lap_u = (uxx + uyy).reshape(nelts, npts, order="F")

        elif dim == 3:
            ur = np.einsum("il,eljk->eijk", d, u_g)
            us = np.einsum("jl,eilk->eijk", d, u_g)
            ut = np.einsum("kl,eijl->eijk", d, u_g)

            ux = g[0,0]*ur + g[0,1]*us + g[0,2]*ut
            uy = g[1,0]*ur + g[1,1]*us + g[1,2]*ut
            uz = g[2,0]*ur + g[2,1]*us + g[2,2]*ut

            uxx = np.einsum("li,eljk->eijk", d, ux)
            uyy = np.einsum("lj,eilk->eijk", d, uy)
            uzz = np.einsum("lk,eijl->eijk", d, uz)

            lap_u = (uxx + uyy + uzz).reshape(nelts, npts, order="F")

        else:
            raise ValueError("Only supported for dim = 2, 3")


        return discr.scatter(discr.apply_mask(lap_u))

    return matvec


def mass_matvec(queue, discr):
    pass


def grad_matvec(queue, discr):
    pass


def div_matvec(queue, discr):
    pass
