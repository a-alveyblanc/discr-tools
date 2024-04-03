"""
Contains routines useful for making sense of reference vs physical space, i.e.
mapping coordinates from reference to physical space and computing the Jacobian
of the inverse mapping.
"""

import numpy as np
import numpy.linalg as la

from discretization.nodal import EquispacedNodalBasis
from discretization.operators import ReferenceOperators


def get_interpolatory_map(ref_nodes):
    """
    Evaluates a 1D Lagrange basis at the reference nodes. This is used to map
    the reference nodes to a physical element by using only the four vertices of 
    the element.
    """
    mapping_cls = EquispacedNodalBasis(1)
    mapping_basis = mapping_cls.basis

    return np.array([ psi(ref_nodes) for psi in mapping_basis ])


def map_elements(ref_nodes, elements):
    """
    Maps reference nodes to each element in the physical space using the map
    returned by `get_interpolatory_map`
    """

    dim, nelts, _ = elements.shape

    interp_map = get_interpolatory_map(ref_nodes)

    return np.einsum(
        "xers,ri,sj->xeij",
        elements.reshape(dim, nelts, *(dim,)*dim),
        interp_map,
        interp_map
    ).reshape(dim, nelts, len(ref_nodes)**dim)


def jacobian(mapped_elements, basis_cls):
    """
    Returns the Jacobian of the mapping using the mapped nodes
    """
    ops = ReferenceOperators(basis_cls)

    dim, nelts, ndofs = mapped_elements.shape
    ndofs_1d = int(ndofs**(1/dim))

    mapped_tp = mapped_elements.reshape(dim, nelts, *(ndofs_1d,)*dim, order="F")

    return np.array([
        np.einsum("il,xelj->xeij", ops.diff_operator, mapped_tp),
        np.einsum("jl,xeil->xeij", ops.diff_operator, mapped_tp),
    ]).reshape(dim, dim, nelts, ndofs, order="F")


def inverse_jacobian_t(mapped_elements, basis_cls):
    """
    Returns the inverse of the mapping Jacobian 
    """
    return la.inv(jacobian(mapped_elements, basis_cls).T).T


def jacobian_determinant(mapped_elements, basis_cls):
    """
    Returns the determinant of the Jacobian of the mapping
    """
    jac = jacobian(mapped_elements, basis_cls)

    det = jac[0,0]*jac[1,1] - jac[0,1]*jac[1,0]

    return np.abs(det)
