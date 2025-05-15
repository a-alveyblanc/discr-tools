import discr_tools.geometry as geo
import discr_tools.matvecs as mv
from discr_tools.assembly import assemble
from discr_tools.discretization import Discretization

import numpy as np

import pyopencl as cl

import scipy.sparse.linalg as spla
import sympy as sp

import time

from discr_tools.nodal import EquispacedNodalBasis


def main(
        nelts_1d,
        order,
        dim,
        direct_solve=False,
        cg_solve=False,
        use_gpu=False,
        visualize=False,
    ):

    if not cg_solve and not direct_solve:
        cg_solve = True

    if dim == 1:
        raise ValueError("1D not supported")

    nelements = nelts_1d**dim
    ndofs = (nelts_1d**dim)*(order+1)**dim
    print(f"Dim        = {dim}")
    print(f"Elements   = {nelements}")
    print(f"Order      = {order}")
    print(f"Total DOFs = {ndofs}")

    x_sp = sp.symbols("".join(f"x{i} " for i in range(dim)))

    u_expr = 1.
    for i in range(dim):
        u_expr *= sp.sin(sp.pi*x_sp[i])  # type: ignore
    u_lambda = sp.lambdify(x_sp, u_expr)

    lap_u_expr = 0.
    for i in range(dim):
        lap_u_expr += u_expr.diff(x_sp[0], 2)  # type: ignore
    lap_u_expr = -lap_u_expr
    lap_u_lambda = sp.lambdify(x_sp, lap_u_expr)

    def u(x):
        return u_lambda(*x)

    def rhs(x):
        return lap_u_lambda(*x)

    a, b = -1, 1
    discr = Discretization(order, a, b, dim, nelts_1d)

    det_j = geo.jacobian_determinant(discr.mapped_elements, discr.basis_cls)
    wts = np.kron(*(discr.basis_cls.weights,)*2)  # type: ignore

    if dim == 3:
        wts = np.kron(wts, discr.basis_cls.weights)

    u_l_exact = u(discr.mapped_elements)
    u_l_l2 = np.sqrt(np.sum(u_l_exact**2 * det_j * wts))

    # direct solver
    if direct_solve:
        start = time.time()
        s_g, rhs_g = assemble(discr, rhs)
        assembly_time = time.time() - start

        start = time.time()
        u_g = spla.spsolve(s_g, rhs_g)
        direct_time = time.time() - start

        u_l = discr.gather(u_g)
        abs_err = np.abs(u_l - u_l_exact)
        direct_rel_l2_err = np.sqrt(np.sum(abs_err**2 * det_j * wts)) / u_l_l2
        print(f"Direct: {direct_rel_l2_err:.3e}, solve took {direct_time:.3f}s",
              f" assembly took {assembly_time:.3f}s")
        print(f"        {(ndofs / direct_time):.3f} DOFs/s")

        if visualize:
            from discr_tools.visualization import plot_solution
            plot_solution(discr, u_l, fig_name="direct.png")

    # CG
    if cg_solve:
        f = det_j * rhs(discr.mapped_elements) * wts
        f_g = discr.scatter(discr.apply_mask(f))

        if use_gpu:
            ctx = cl.create_some_context()
            queue = cl.CommandQueue(ctx)

            matvec = mv.PoissonMatvec(discr, queue=queue)
        else:
            matvec = mv.PoissonMatvec(discr)

        lin_op = spla.LinearOperator(f_g.shape*2, matvec)

        start = time.time()
        u_matfree, _ = spla.cg(lin_op, f_g)
        it_time = time.time() - start

        u_l_matfree = discr.gather(u_matfree)

        abs_err = np.abs(u_l_matfree - u_l_exact)
        it_rel_l2_err = np.sqrt(np.sum(abs_err**2 * det_j * wts)) / u_l_l2
        print(f"CG    : {it_rel_l2_err:.3e}, solve took {it_time:.3f} s")
        print(f"        {(ndofs / it_time):.3f} DOFs/s")

        if visualize:
            from discr_tools.visualization import plot_contourf
            plot_contourf(u_l_matfree, discr)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--nelts_1d", action="store", type=int,
                        default=10)
    parser.add_argument("--order", action="store", type=int,
                        default=5)
    parser.add_argument("--dim", action="store", type=int,
                        default=2)
    parser.add_argument("--direct_solve", action="store_true")
    parser.add_argument("--use_cg", action="store_true")
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--visualize", action="store_true")

    args = parser.parse_args()

    main(
        nelts_1d=args.nelts_1d,
        order=args.order,
        dim=args.dim,
        direct_solve=args.direct_solve,
        cg_solve=args.use_cg,
        use_gpu=args.use_gpu,
        visualize=args.visualize
    )
