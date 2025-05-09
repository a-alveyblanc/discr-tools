import discr_tools.geometry as geo
import discr_tools.kernels as knl
from discr_tools.assembly import assemble
from discr_tools.discretization import Discretization

import os

import numpy as np
import numpy.linalg as la

import scipy.sparse.linalg as spla
import sympy as sp

import time


if "PYOPENCL_CTX" not in os.environ:
    os.environ["PYOPENCL_CTX"] = "1"


def main(nelts_1d, order, dim, direct_solve, iterative_solve):
    x_sp = sp.symbols('x0 x1')

    u_expr = sp.sin(sp.pi*x_sp[0])*sp.sin(sp.pi*x_sp[1])  # type: ignore
    u_lambda = sp.lambdify(x_sp, u_expr)

    lap_u_expr = -(u_expr.diff(x_sp[0], 2) + u_expr.diff(x_sp[1], 2))
    lap_u_lambda = sp.lambdify(x_sp, lap_u_expr)

    def u(x):
        x, y = x
        return u_lambda(x, y)

    def rhs(x):
        x, y = x
        return lap_u_lambda(x, y)

    a, b = -1, 1
    discr = Discretization(order, a, b, dim, nelts_1d)

    inv_jac_t = geo.inverse_jacobian_t(discr.mapped_elements, discr.basis_cls)
    det_j = geo.jacobian_determinant(discr.mapped_elements, discr.basis_cls)
    wts_2d = np.kron(*(discr.basis_cls.weights,)*2)  # type: ignore
    g = np.einsum("kiep,kjep,ep,p->ijep", inv_jac_t, inv_jac_t, det_j, wts_2d)

    def matvec(u):
        u_g = discr.gather(u)

        eye = np.eye(order+1)
        dr = np.kron(discr.operators.diff_operator, eye)
        ds = np.kron(eye, discr.operators.diff_operator)

        ur = np.einsum("il,el->ei", dr, u_g)
        us = np.einsum("il,el->ei", ds, u_g)

        ux = g[0,0]*ur + g[0,1]*us
        uy = g[1,0]*ur + g[1,1]*us

        uxx = np.einsum("li,el->ei", dr, ux)
        uyy = np.einsum("li,el->ei", ds, uy)

        lap_u = discr.apply_boundary_condition(uxx + uyy)

        return discr.scatter(lap_u)

    u_l_exact = u(discr.mapped_elements)
    u_l_l2 = np.sqrt(np.sum(u_l_exact**2 * det_j * wts_2d))

    # direct solver
    if direct_solve:
        start = time.time()
        s_g, rhs_g = assemble(discr, rhs)
        u_g = spla.spsolve(s_g, rhs_g)
        direct_time = time.time() - start

        u_l = discr.gather(u_g)
        abs_err = np.abs(u_l - u_l_exact)
        direct_rel_l2_err = np.sqrt(np.sum(abs_err**2 * det_j * wts_2d)) / u_l_l2
        print(f"Direct: {direct_rel_l2_err:.3e}, took {direct_time:.3f} s")

    if iterative_solve:
        # iterative solver (CG for now)
        f = det_j * rhs(discr.mapped_elements) * wts_2d
        f_g = discr.scatter(discr.apply_boundary_condition(f))

        lin_op = spla.LinearOperator(f_g.shape*2, matvec)

        start = time.time()
        u_matfree, _ = spla.cg(lin_op, f_g)
        it_time = time.time() - start

        u_l_matfree = discr.gather(u_matfree)

        abs_err = np.abs(u_l_matfree - u_l_exact)
        it_rel_l2_err = np.sqrt(np.sum(abs_err**2 * det_j * wts_2d)) / u_l_l2
        print(f"CG    : {it_rel_l2_err:.3e}, took {it_time:.3f} s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--nelts_1d", action="store", type=int,
                        default=4)
    parser.add_argument("--order", action="store", type=int,
                        default=1)
    parser.add_argument("--dim", action="store", type=int,
                        default=2)
    parser.add_argument("--direct_solve", action="store_true")
    parser.add_argument("--iterative_solve", action="store_true")

    args = parser.parse_args()

    main(
        nelts_1d=args.nelts_1d,
        order=args.order,
        dim=args.dim,
        direct_solve=args.direct_solve,
        iterative_solve=args.iterative_solve
    )
