import numpy as np
import numpy.linalg as la

from discr_tools.nodal import EquispacedNodalBasis
from discr_tools.operators import ReferenceOperators


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

    if dim == 1:
        return np.einsum(
            "xer,ri->xei",
            elements.reshape(dim, nelts, *(2,)*dim, order="F")
        )
    elif dim == 2:
        return np.einsum(
            "xers,ri,sj->xeij",
            elements.reshape(dim, nelts, *(2,)*dim, order="F"),
            interp_map,
            interp_map
        ).reshape(dim, nelts, len(ref_nodes)**dim, order="F")
    elif dim == 3:
        return np.einsum(
            "xerst,ri,sj,tk->xeijk",
            elements.reshape(dim, nelts, *(2,)*dim, order="F"),
            interp_map,
            interp_map,
            interp_map
        ).reshape(dim, nelts, len(ref_nodes)**dim, order="F")
    else:
        raise NotImplementedError("Only implemented for 2 <= dim <= 3")


def jacobian(mapped_elements, basis_cls):
    """
    Returns the Jacobian of the mapping using the mapped nodes
    """
    ops = ReferenceOperators(basis_cls)

    dim, nelts, ndofs = mapped_elements.shape
    ndofs_1d = len(basis_cls.nodes)

    mapped_tp = mapped_elements.reshape(dim, nelts, *(ndofs_1d,)*dim, order="F")

    if dim == 2:
        return np.array([
            np.einsum("il,xelj->xeij", ops.diff_operator, mapped_tp),  # dx/dr
            np.einsum("jl,xeil->xeij", ops.diff_operator, mapped_tp),  # dx/ds
        ]).reshape(dim, dim, nelts, ndofs, order="F")
    elif dim == 3:
        return np.array([
            np.einsum("il,xeljk->xeijk", ops.diff_operator, mapped_tp), # dx/dr
            np.einsum("jl,xeilk->xeijk", ops.diff_operator, mapped_tp), # dx/ds
            np.einsum("kl,xeijl->xeijk", ops.diff_operator, mapped_tp), # dx/dt
        ]).reshape(dim, dim, nelts, ndofs, order="F")
    else:
        raise NotImplementedError("Only implemented for 2 <= dim <= 3")


def inverse_jacobian_t(mapped_elements, basis_cls):
    """
    Returns the inverse of the mapping Jacobian
    """
    jac = np.moveaxis(jacobian(mapped_elements, basis_cls), (2, 3), (0, 1))
    inv_jac = np.moveaxis(la.inv(jac), (0, 1), (2, 3))
    inv_jac_t = np.transpose(inv_jac, axes=(1, 0, 2, 3))

    return inv_jac_t


def jacobian_determinant(mapped_elements, basis_cls):
    """
    Returns the determinant of the Jacobian of the mapping
    """
    jac = jacobian(mapped_elements, basis_cls)

    return la.det(jac.T).T
