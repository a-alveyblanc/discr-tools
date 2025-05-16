from typing import Callable
import discr_tools.geometry as geo
import discr_tools.kernels as knl
import discr_tools.matvecs as mv
from discr_tools.assembly import assemble
from discr_tools.discretization import Discretization

import numpy as np
import numpy.linalg as la

import os

import pyopencl as cl

import scipy.sparse.linalg as spla
import sympy as sp

import time


class AdvectionMatvec(mv.MatvecBase):
    def __init__(self, alpha, discr, queue=None):
        self.alpha = alpha
        super().__init__(discr, queue=queue)

    def gpu_matvec(self) -> Callable:
        raise NotImplementedError

    def cpu_matvec(self) -> Callable:
        v = self.discr.operators.vandermonde
        vp = self.discr.operators.vandermonde_dx
        wts = self.discr.basis_cls.weights

        det_j = geo.jacobian_determinant(
            self.discr.mapped_elements,
            self.discr.basis_cls
        )

        inv_jac_t = geo.inverse_jacobian_t(
            self.discr.mapped_elements,
            self.discr.basis_cls
        )

        dim, _, npts = self.discr.mapped_elements.shape

        if dim == 2:
            wts = np.kron(wts, wts).flatten()

            v = np.kron(v, v)

            npts_1d = int(np.sqrt(npts))
            eye = np.eye(npts_1d)
            vp_r = np.kron(vp, eye)
            vp_s = np.kron(eye, vp)
            vp = np.array([vp_r, vp_s])
        elif dim == 3:
            npts_1d = int(np.cbrt(npts))
            wts = np.kron(wts, np.kron(wts, wts)).flatten()

            v = np.kron(v, np.kron(v, v))

            eye = np.eye(npts_1d)
            vp_r = np.kron(vp, np.kron(eye, eye))
            vp_s = np.kron(eye, np.kron(vp, eye))
            vp_t = np.kron(eye, np.kron(eye, vp))
            vp = np.array([vp_r, vp_s, vp_t])
        else:
            raise ValueError("Only supports dim = 2, 3")

        g = np.einsum("xrek,ek->xrek", inv_jac_t, det_j)
        m_inv = la.inv(np.diag(wts))

        def matvec(u):
            u = self.discr.gather(u)

            adv_u = np.einsum(
                "ij,x,rki,xrek,k,kj,ej->ei",
                m_inv, self.alpha, vp, g, wts, v, u
            )

            return self.discr.scatter(self.discr.apply_mask(adv_u))

        return matvec


def backward_euler_step(u, matvec, dt):
    def step(uk):
        return uk - dt * matvec(uk)

    lin_op = spla.LinearOperator(u.shape*2, step)
    sol, _ = spla.cg(lin_op, u)

    return sol


def forward_euler_step(u, matvec, dt):
    return dt * matvec(u)


def main(
        nelts_1d,
        order,
        use_gpu=False,
        visualize=False,
    ):

    a, b = -1, 1
    dim = 2

    discr = Discretization(order, a, b, dim, nelts_1d)
    alpha = np.array([1.0, 1.0])

    def u_init(x):
        return np.exp(
            -(x[0]**2 + x[1]**2) / 0.5
        )

    adv_matvec = AdvectionMatvec(alpha, discr)

    nsteps = 100
    u0 = discr.scatter(u_init(discr.mapped_elements))
    u = np.zeros((nsteps, *u0.shape))
    u[0] = u0.copy()
    for istep in range(1,nsteps):
        u[istep] = backward_euler_step(
            u[istep-1], adv_matvec, .1)

    print(la.norm(u[0] - u[-1]))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--nelts_1d", action="store", type=int,
                        default=10)
    parser.add_argument("--order", action="store", type=int,
                        default=5)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--visualize", action="store_true")

    args = parser.parse_args()

    main(
        nelts_1d=args.nelts_1d,
        order=args.order,
        use_gpu=args.use_gpu,
        visualize=args.visualize
    )
