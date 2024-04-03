"""
This file contains routines to generate uniform rectangular meshes
"""

import numpy as np


class Mesh:
    def __init__(self, a, b, dim, nelts):
        self.a = a
        self.b = b
        self.dim = dim
        self.nelts = nelts

    
    def _mesh(self):
        """
        Generates a uniform rectangular mesh
        """

        # generate the grid
        omega_1d = np.linspace(-1, 1, num=self.nelts)
        x, y = np.meshgrid(omega_1d, omega_1d, indexing="ij")

        # number elements starting from bottom left, nodes ccw

    @property
    def mesh(self):
        try:
            return self.__mesh
        except AttributeError:
            self.__mesh = self._mesh()
            return self.__mesh
