import numpy as np
import numpy.linalg as la

import matplotlib.pyplot as plt

from functools import cmp_to_key

from discr_tools import geometry as geo
from discr_tools.nodal import GLLNodalBasis
from discr_tools.operators import ReferenceOperators
from discr_tools.mesh import Mesh


class Discretization:

    def __init__(self, order, a, b, dim, nelts_1d, basis_cls=GLLNodalBasis):

        self.dim = dim
        self.order = order
        self.nelts = nelts_1d**dim
        self.nelts_1d = nelts_1d
        self.interval = [a, b]

        self._basis_cls = basis_cls(order)
        self._operators = ReferenceOperators(self._basis_cls)

        self._mesh = Mesh(a, b, dim, nelts_1d)
        self._mapped_elements = geo.map_elements(self._basis_cls.nodes,
                                                 self.mesh.elements)

        self.__local_to_global = self._local_to_global()

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
    def local_to_global(self):
        try:
            return self.__local_to_global
        except AttributeError:
            self.__local_to_global = self._local_to_global()
            return self.__local_to_global

    @property
    def basis_cls(self):
        return self._basis_cls

    def _local_to_global(self):
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

            for i in range(self.dim):
                if a[i] > b[i]:
                    return 1
                if a[i] < b[i]:
                    return -1

            return 0

        local_ordering = sorted(local_ordering,
                                key=cmp_to_key(lexicographical_sort))

        # }}}

        # {{{ construct new node set and global ids

        tol = 1e-14
        global_id = 0
        global_to_local = []
        new_nodes = []
        total_nodes = nnodes * nelts
        for i in range(total_nodes-1):
            global_to_local.append((global_id, local_ordering[i]))

            pt1 = local_ordering[i][1]
            pt2 = local_ordering[i+1][1]
            if la.norm(pt1 - pt2) > tol:
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

        self.__local_to_global = global_indices
        self._nodes = new_nodes

        return global_indices

    def _boundary_condition_indices(self):
        """
        Construct an indexing mask to apply to an operator so boundary
        conditions can be implemented
        """
        nodes = self.nodes
        dim = nodes.shape[0]
        dim, _ = nodes.shape

        tol = 1e-12

        if dim == 2:

            x, y = nodes

            x_max = np.max(x)
            x_min = np.min(x)

            y_max = np.max(y)
            y_min = np.min(y)

            x_max_idxs = np.where(np.abs(x - x_max) < tol)
            x_min_idxs = np.where(np.abs(x - x_min) < tol)

            y_max_idxs = np.where(np.abs(y - y_max) < tol)
            y_min_idxs = np.where(np.abs(y - y_min) < tol)

            boundary_idxs = np.unique(np.vstack([
                x_min_idxs, x_max_idxs,
                y_min_idxs, y_max_idxs,
            ])).flatten()

        elif dim == 3:

            x, y, z = nodes

            x_max = np.max(x)
            x_min = np.min(x)

            y_max = np.max(y)
            y_min = np.min(y)

            z_max = np.max(z)
            z_min = np.min(z)

            x_max_idxs = np.where(np.abs(x - x_max) < tol)
            x_min_idxs = np.where(np.abs(x - x_min) < tol)

            y_max_idxs = np.where(np.abs(y - y_max) < tol)
            y_min_idxs = np.where(np.abs(y - y_min) < tol)

            z_max_idxs = np.where(np.abs(z - z_max) < tol)
            z_min_idxs = np.where(np.abs(z - z_min) < tol)

            boundary_idxs = np.unique(np.vstack([
                x_min_idxs, x_max_idxs,
                y_min_idxs, y_max_idxs,
                z_min_idxs, z_max_idxs
            ])).flatten()

        else:
            raise NotImplementedError("Only implemented for dim = 2, 3")

        return boundary_idxs

    def apply_mask(self, vec):
        """
        Apply a boundary mask to a vector of DOF data or a matrix.

        Enforces homogeneous boundary conditions
        """
        idxs = self._boundary_condition_indices()

        # vector case (used for cg)
        if isinstance(vec, np.ndarray):
            _, nnodes = self.nodes.shape
            _, nelts, _ = self.mapped_elements.shape

            bc_mask = np.ones((nnodes))
            bc_mask[idxs] = 0.0

            g2l = self.local_to_global
            for ielt in range(nelts):
                vec[ielt,:] = vec[ielt,:] * bc_mask[g2l[ielt]]

        # sparse matrix case (used for direct solves)
        else:
            bdry_flags = np.zeros(len(vec.data), dtype=bool)
            bdry_flags[idxs] = True

            for ell, (i,j) in enumerate(zip(vec.row, vec.col)):
                if bdry_flags[i] or bdry_flags[j]:
                    if i == j:
                        vec.data[ell] = 1.0
                    else:
                        vec.data[ell] = 0.0

        return vec

    def gather(self, vec):
        nelts, npts = self.local_to_global.shape
        vec_g = np.zeros((nelts, npts))

        for ielt in range(nelts):
            vec_g[ielt] = vec[self.local_to_global[ielt]]

        return vec_g

    def scatter(self, vec):
        nelts, _ = self.local_to_global.shape
        vec_s = np.zeros(len(np.unique(self.local_to_global.flatten())))

        for ielt in range(nelts):
            vec_s[self.local_to_global[ielt]] += vec[ielt]

        return vec_s

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
