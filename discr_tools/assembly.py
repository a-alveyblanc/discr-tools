"""
This file contains routines for assembling the global system
"""
import numpy as np
import numpy.linalg as la

import scipy.sparse as sp

import discr_tools.geometry as geo


def assemble(discr, rhs):
    """
    Using a discretization and right-hand side, construct the global stiffness
    matrix and load vector.

    The stiffness matrix is returned as a CSR array.
    """
    # get nodes, mesh size
    x = discr.mapped_elements

    # get operators
    op = discr.operators
    Vp_1d = op.vandermonde_dx
    I_1d = np.eye(Vp_1d.shape[0])  # used to construct 2D operators

    # get quadrature weights and |J|
    wts = discr.basis_cls.weights
    det_j = geo.jacobian_determinant(x, discr.basis_cls)
    inv_jac_t = geo.inverse_jacobian_t(x, discr.basis_cls)
    g = np.einsum(
        "kiep,kjep,ep,p->ijep",
        inv_jac_t,
        inv_jac_t,
        det_j,
        np.kron(wts, wts)
    )

    # evaluate rhs for all elements
    f = det_j * rhs(x) * np.kron(wts, wts).flatten()

    # form reference stiffness matrix (uniform mesh -> never changes)
    Dr = np.kron(Vp_1d, I_1d)
    Ds = np.kron(I_1d, Vp_1d)
    D = np.array([Dr, Ds])

    # NOTE: forms all element-local operators; not advised in practice
    S_local = np.einsum("rki,xrek,rkj->eij", D, g, D)

    # assembly
    # sparse global stiffness matrix & load vector
    _, nelts, ndofs = x.shape

    S_data = np.zeros((nelts, ndofs**2))
    S_rows = np.zeros_like(S_data).astype(int)
    S_cols = np.zeros_like(S_data).astype(int)

    f_data = np.zeros((nelts,ndofs))
    f_rows = np.zeros_like(f_data).astype(int)
    f_cols = np.zeros_like(f_data).astype(int)

    g2l = discr.local_to_global # global to local map
    for ielt in range(nelts):
        # rhs
        f_data[ielt] = f[ielt]
        f_rows[ielt] = g2l[ielt]
        f_cols[ielt] = 0

        # global stiffness matrix
        S_data[ielt] = S_local[ielt].flatten()
        S_rows[ielt] = np.repeat(g2l[ielt], ndofs)
        S_cols[ielt] = np.tile(g2l[ielt], ndofs)

    # assemble global stiffness + apply boundary conditions
    S_coordinates = (S_rows.flatten(), S_cols.flatten())
    S_global = sp.coo_matrix((S_data.flatten(), S_coordinates))
    S_global.sum_duplicates()

    S_global = discr.apply_boundary_condition(S_global).tocsr()

    # assemble global load vector
    f_coordinates = (f_rows.flatten(), f_cols.flatten())
    f_global = sp.coo_matrix((f_data.flatten(), f_coordinates)).toarray()

    return S_global, f_global
