import discr_tools.geometry as geo
import discr_tools.kernels as knl
from discr_tools.discretization import Discretization

import numpy as np
import numpy.linalg as la

import pytest

import pyopencl as cl


@pytest.mark.parametrize("order", [4, 6, 8, 10])
@pytest.mark.parametrize("nelts_1d", [1, 2, 4, 8])
def test_gradient_3d(order, nelts_1d):
    a, b = -1, 1
    dim = 3

    discr = Discretization(order, a, b, dim, nelts_1d)
    x = discr.mapped_elements
    basis = discr.basis_cls

    f = x[0] + x[1]**2 + x[2]**3
    grad_f = np.array([
        1 + 0 * x[0],
        2*x[1],
        3*x[2]**2
    ])

    g = geo.inverse_jacobian_t(x, basis)

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    grad = knl.gradient_3d(queue, discr.operators.diff_operator, f, g)
    grad = grad.reshape(3, -1, (order+1)**3, order="F").get()

    grad = grad.flatten()
    grad_f = grad_f.flatten()
    err = la.norm(grad_f - grad, np.inf) / la.norm(grad, np.inf)

    assert err <= 1e-11


@pytest.mark.parametrize("order", [4, 6, 8, 10])
@pytest.mark.parametrize("nelts_1d", [1, 2, 4, 8])
def test_divergence_3d(order, nelts_1d):
    a, b = -1, 1
    dim = 3

    discr = Discretization(order, a, b, dim, nelts_1d)
    x = discr.mapped_elements
    basis = discr.basis_cls

    f = np.array([
        x[0],
        x[1]**2,
        x[2]**3
    ])

    div_f = (1 + 0*x[0]) + 2*x[1] + 3*x[2]**2

    g = geo.inverse_jacobian_t(x, basis)

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    div = knl.divergence_3d(queue, discr.operators.diff_operator, f, g)
    div = div.reshape(-1, (order+1)**3, order="F").get()

    div = div.flatten()
    div_f = div_f.flatten()
    err = la.norm(div_f - div, np.inf) / la.norm(div, np.inf)

    assert err <= 1e-10
