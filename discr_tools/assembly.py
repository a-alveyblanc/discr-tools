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
    h = np.abs(x[0,0,0] - x[0,0,-1]) # uniform mesh -> never changes

    # get operators
    op = discr.operators
    M_1d = op.mass_matrix
    Vp_1d = op.vandermonde_dx
    I_1d = np.eye(Vp_1d.shape[0]) # used to construct 2D operators

    # get quadrature weights and |J|
    wts = discr.basis_cls.weights
    det_J = geo.jacobian_determinant(x, discr.basis_cls)

    # evaluate rhs for all elements
    f = det_J * rhs(x) * np.kron(wts, wts).flatten()

    # form reference stiffness matrix (uniform mesh -> never changes)
    W = (h/2)*np.kron(M_1d, M_1d)
    K = (2/h)*la.inv(M_1d) @ Vp_1d.T @ M_1d @ Vp_1d
    S_local = W @ (np.kron(K, I_1d) + np.kron(I_1d, K))

    # assembly
    # sparse global stiffness matrix & load vector
    _, nelts, ndofs = x.shape

    S_data = np.zeros((nelts, ndofs**2))
    S_rows = np.zeros_like(S_data).astype(int)
    S_cols = np.zeros_like(S_data).astype(int)

    f_data = np.zeros((nelts,ndofs))
    f_rows = np.zeros_like(f_data).astype(int)
    f_cols = np.zeros_like(f_data).astype(int)

    g2l = discr.global_to_local # global to local map
    for ielt in range(nelts):
        # rhs
        f_data[ielt] = f[ielt]
        f_rows[ielt] = g2l[ielt]
        f_cols[ielt] = 0

        # global stiffness matrix
        S_data[ielt] = S_local.flatten()
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
