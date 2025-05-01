import discr_tools.geometry as geo
import discr_tools.kernels as knl
from discr_tools.discretization import Discretization

import numpy as np
import numpy.linalg as la

import pyopencl as cl
import pytest
from pytools.convergence import EOCRecorder

# TODO: use sympy to come up with better test cases?


@pytest.mark.parametrize("order", [2, 4, 8])
def test_gradient_3d(order):
    a, b = -1, 1
    dim = 3

    eoc_rec = EOCRecorder()

    for nelts_1d in [4, 8, 10]:
        discr = Discretization(order, a, b, dim, nelts_1d)
        x = discr.mapped_elements
        basis = discr.basis_cls

        f = np.cos(x[0])*np.sin(x[1])*np.cos(x[2])
        grad_f = np.array([
            -np.sin(x[0])*np.sin(x[1])*np.cos(x[2]),
            np.cos(x[0])*np.cos(x[1])*np.cos(x[2]),
            -np.cos(x[0])*np.sin(x[1])*np.sin(x[2])
        ])

        g = geo.inverse_jacobian_t(x, basis)

        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)

        grad = knl.gradient_3d(queue, discr.operators.diff_operator, f, g)
        grad = grad.reshape(3, -1, (order+1)**3, order="F")

        grad = grad.flatten()
        grad_f = grad_f.flatten()
        err = la.norm(grad_f - grad, np.inf) / la.norm(grad, np.inf)
        eoc_rec.add_data_point(1.0/nelts_1d, err)

    print(eoc_rec)
    assert (
        eoc_rec.order_estimate() > (order - 0.5) or eoc_rec.max_error() < 1e-11
    )


@pytest.mark.parametrize("order", [2, 4, 8])
def test_divergence_3d(order):
    a, b = -1, 1
    dim = 3

    eoc_rec = EOCRecorder()

    for nelts_1d in [4, 6, 8]:
        discr = Discretization(order, a, b, dim, nelts_1d)
        x = discr.mapped_elements
        basis = discr.basis_cls

        f = np.array([
            np.sin(x[0])*np.cos(x[1]),
            np.cos(x[1])*np.sin(x[2]),
            np.sin(x[0])*np.cos(x[1])*np.sin(x[2])
        ])

        div_f = np.cos(x[0])*np.cos(x[1]) + \
               -np.sin(x[1])*np.sin(x[2]) + \
                np.sin(x[0])*np.cos(x[1])*np.cos(x[2])

        g = geo.inverse_jacobian_t(x, basis)

        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)

        div = knl.divergence_3d(queue, discr.operators.diff_operator, f, g)
        div = div.reshape(-1, (order+1)**3, order="F")

        div = div.flatten()
        div_f = div_f.flatten()
        err = la.norm(div_f - div, np.inf) / la.norm(div, np.inf)
        eoc_rec.add_data_point(1/nelts_1d, err)

    print(eoc_rec)
    assert (
        eoc_rec.order_estimate() >= (order - 0.5) or eoc_rec.max_error() < 1e-10
    )


@pytest.mark.parametrize("order", [2, 4, 8])
def test_divgrad_3d(order):
    a, b = -1, 1
    dim = 3

    eoc_rec = EOCRecorder()

    for nelts_1d in [8, 12, 16]:
        discr = Discretization(order, a, b, dim, nelts_1d)
        x = discr.mapped_elements
        basis = discr.basis_cls

        f = np.cos(x[0])*np.sin(x[1])*np.cos(x[2])
        divgrad_f = -3*np.cos(x[0])*np.sin(x[1])*np.cos(x[2])

        g = geo.inverse_jacobian_t(x, basis)

        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)

        divgrad = knl.divgrad_3d(queue, discr.operators.diff_operator, f, g)
        divgrad = divgrad.reshape(-1, (order+1)**3, order="F")

        divgrad = divgrad.flatten()
        divgrad_f = divgrad_f.flatten()
        err = la.norm(divgrad_f - divgrad, np.inf) / la.norm(divgrad, np.inf)
        eoc_rec.add_data_point(1/nelts_1d, err)

    # order requirements could be more strict, but we're just looking for
    # something comparable to the analytic solution
    print(eoc_rec)
    assert (
        eoc_rec.order_estimate() >= (order - 0.5) or eoc_rec.max_error() < 1e-5
    )

