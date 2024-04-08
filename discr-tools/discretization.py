"""
This file contains routines necessary for constructing a full discretization.
Specifically, this combines all other files into a single class so that the
mesh, operators, basis, etc., are all easily accessible.
"""

import numpy as np

import matplotlib.pyplot as plt

from . import geometry as geo
from .nodal import GLLNodalBasis
from .operators import ReferenceOperators
from .mesh import Mesh


class Discretization:

    def __init__(self, order, a, b, dim, nelts_1d, basis_cls=GLLNodalBasis):

        self._basis_cls = basis_cls(order)
        self._operators = ReferenceOperators(self._basis_cls)

        self._mesh = Mesh(a, b, dim, nelts_1d)
        self._mapped_elements = geo.map_elements(self._basis_cls.nodes,
                                                 self.mesh.elements)

        dim = self._mapped_elements.shape[0]
        self._nodes = self._mapped_elements.reshape(dim, -1)


    @property
    def mesh(self):
        return self._mesh
    

    @property
    def operators(self):
        return self._operators


    @property
    def mapped_elements(self):
        return self._mapped_elements
    

    @property
    def basis_cls(self):
        return self._basis_cls


    @property
    def nodes(self):
        return self._nodes


    def _boundary_condition_mask(self, nnz):
        """
        Construct an indexing mask to apply to an operator so boundary
        conditions can be implemented
        """
        nodes = self.nodes
        dim, _ = nodes.shape

        tol = 1e-12

        if dim == 2:
            
            x, y = nodes
            
            # {{{ find boundary nodes and indices

            x_min = np.min(x)
            x_max = np.max(x)

            y_min = np.min(y)
            y_max = np.max(y)

            x_min_idx = np.where(np.abs(x - x_min) < tol)
            x_max_idx = np.where(np.abs(x - x_max) < tol)

            y_min_idx = np.where(np.abs(y - y_min) < tol)
            y_max_idx = np.where(np.abs(y - y_max) < tol)

            all_idxs = np.vstack([
                x_min_idx, x_max_idx, y_min_idx, y_max_idx
            ]).flatten()

            # }}}

            # {{{ construct mask

            bc_mask = np.zeros(nnz, dtype=bool)
            bc_mask[all_idxs] = True

            # }}}

        elif dim == 3:

            x, y, z = nodes

            # {{{ find boundary nodes and indices

            x_min = np.min(x)
            x_max = np.max(x)

            y_min = np.min(y)
            y_max = np.max(y)

            z_min = np.min(z)
            z_max = np.max(z)

            x_min_idx = np.where(np.abs(x - x_min) < tol)
            x_max_idx = np.where(np.abs(x - x_max) < tol)

            y_min_idx = np.where(np.abs(y - y_min) < tol)
            y_max_idx = np.where(np.abs(y - y_max) < tol)

            z_min_idx = np.where(np.abs(z - z_min) < tol)
            z_max_idx = np.where(np.abs(z - z_max) < tol)

            all_idxs = np.unique(np.vstack([
                x_min_idx, x_max_idx, 
                y_min_idx, y_max_idx, 
                z_min_idx, z_max_idx
            ])).flatten()

            # }}}

            # {{{ construct mask

            bc_mask = np.zeros(nnz, dtype=bool)
            bc_mask[all_idxs] = True

            # }}}

        else:
            raise NotImplementedError("Only implemented for 2 <= dim <= 3")

        return bc_mask


    def apply_boundary_condition(self, A):
        """
        Update an operator using `_boundary_condition_mask`.

        `A` is expected to be a sparse matrix.

        Current capability is limited to homogeneous Dirichlet boundary
        conditions.
        """
        bc_mask = self._boundary_condition_mask(len(A.data))

        for ell, (i,j) in enumerate(zip(A.row, A.col)):
            if bc_mask[i] or bc_mask[j]:
                if i == j:
                    A.data[ell] = 1.0
                else:
                    A.data[ell] = 0.0

        return A



    
    def plot_mapped_elements(self):
        plt.plot(self.mesh.vertices[0], self.mesh.vertices[1], '*',
                 c='tab:red', label='Element vertices', zorder=5, markersize=10)

        plt.plot(self.mapped_elements[0].ravel(),
                 self.mapped_elements[1].ravel(),
                 '.', c='tab:blue', label="Nodes")

        plt.grid()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
