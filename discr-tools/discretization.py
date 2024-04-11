"""
This file contains routines necessary for constructing a full discretization.
Specifically, this combines all other files into a single class so that the
mesh, operators, basis, etc., are all easily accessible.
"""

import numpy as np
import numpy.linalg as la

import matplotlib.pyplot as plt

from functools import cmp_to_key

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

        self.__global_to_local = self._global_to_local()

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
    def nodes(self):
        return self._nodes


    @property
    def global_to_local(self):
        try:
            return self.__global_to_local
        except AttributeError:
            self.__global_to_local = self._global_to_local()
            return self.__global_to_local


    @property
    def basis_cls(self):
        return self._basis_cls


    def _global_to_local(self):
        """
        Establishes global to local node numbering for the case where the number
        of vertices is smaller than the total number of nodes in an element
        """
        elts = self.mapped_elements
        _, nelts, nnodes = elts.shape

        # {{{ establish local ordering 

        local_ordering = []
        ipoint = 0
        for ielt in range(nelts):
            for inode in range(nnodes):
                local_ordering.append((ipoint, elts[:,ielt,inode]))
                ipoint += 1

        def lexicographical_sort(a, b):
            """
            Compare two points lexicographically, i.e.

            (0, 0) < (0, 1) < (0, 2) < ... < (0, n) < (1, 0) < ... < (0, 0)
            """

            if isinstance(a, tuple):
                a = a[1]
                b = b[1]
            
            if a[0] > b[0]:
                return 1
            if a[0] < b[0]:
                return -1
            if a[1] > b[1]:
                return 1
            if a[1] < b[1]:
                return -1
            else:
                return 0

        local_ordering = sorted(local_ordering, 
                                key=cmp_to_key(lexicographical_sort))
        
        # }}}

        # {{{ construct new node set and global ids

        node_tol = 1e-14
        global_id = 0
        global_to_local = []
        new_nodes = []
        total_nodes = nnodes * nelts
        for i in range(total_nodes-1):
            global_to_local.append((global_id, local_ordering[i]))

            pt1 = local_ordering[i][1]
            pt2 = local_ordering[i+1][1]
            if la.norm(pt1 - pt2) > node_tol:
                new_nodes.append(pt1)
                global_id += 1

        new_nodes.append(local_ordering[-1][1])
        new_nodes = np.array(new_nodes).T
        global_to_local.append((global_id, local_ordering[-1]))

        # }}}

        # {{{ return to local ordering and snatch global indices

        def local_sort(a, b):
            """
            Return to local ordering
            """
            a = a[1][0]
            b = b[1][0]

            if a > b:
                return 1
            elif a < b:
                return -1
            else:
                return 0

        sorted_elts = sorted(global_to_local, key=cmp_to_key(local_sort))

        ipoint = 0
        global_indices = np.zeros((nelts, nnodes), dtype=int)
        for ielt in range(nelts):
            for inode in range(nnodes):
                global_indices[ielt,inode] = sorted_elts[ipoint][0]
                ipoint += 1

        # }}}

        self.__global_to_local = global_indices
        self._nodes = new_nodes

        return global_indices


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
