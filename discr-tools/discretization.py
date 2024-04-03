"""
This file contains routines necessary for constructing a full discretization.
Specifically, this combines all other files into a single class so that the
mesh, operators, basis, etc., are all easily accessible.
"""

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

    
    def plot_mapped_elements(self):
        import matplotlib.pyplot as plt

        plt.plot(self.mesh.vertices[0], self.mesh.vertices[1], '*',
                 c='tab:red', label='Element vertices', zorder=5, markersize=10)

        plt.plot(self.mapped_elements[0].ravel(),
                 self.mapped_elements[1].ravel(),
                 '.', c='tab:blue', label="Nodes")

        plt.grid()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
