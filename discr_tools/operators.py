import numpy as np
import numpy.linalg as la


class ReferenceOperators:
    """
    Used to easily access necessary reference operators.
    """
    def __init__(self, basis_cls):
        self.basis = basis_cls.basis
        self.basis_dx = basis_cls.basis_dx
        self.nodes = basis_cls.nodes
        self.weights = basis_cls.weights


    # {{{ compute operators

    def _vandermonde(self):
        return np.array([
            self.basis[i](self.nodes)
            for i in range(len(self.basis))
        ]).T

    def _vandermonde_dx(self):
       return np.array([
           self.basis_dx[i](self.nodes)
           for i in range(len(self.basis))
       ]).T

    def _diff_operator(self):
        vdm = self.vandermonde
        vdm_p = self.vandermonde_dx

        return vdm_p @ la.inv(vdm)

    def _mass_matrix(self):
        return np.diag(self.weights)

    def _stiffness_matrix(self):
        return np.einsum(
            "k,ki,kj->ij",
            self.weights,
            *(self.vandermonde_dx,)*2
        )

    # }}}


    # {{{ access operators

    @property
    def vandermonde(self):
        try:
            return self.__vandermonde
        except AttributeError:
            self.__vandermonde = self._vandermonde()
            return self.__vandermonde

    @property
    def vandermonde_dx(self):
        try:
            return self.__vandermonde_dx
        except AttributeError:
            self.__vandermonde_dx = self._vandermonde_dx()
            return self.__vandermonde_dx

    @property
    def diff_operator(self):
        try:
            return self.__diff_operator
        except AttributeError:
            self.__diff_operator = self._diff_operator()
            return self.__diff_operator

    @property
    def mass_matrix(self):
        try:
            return self.__mass_matrix
        except AttributeError:
            self.__mass_matrix = self._mass_matrix()
            return self.__mass_matrix

    @property
    def stiffness_matrix(self):
        try:
            return self.__stiffness_matrix
        except AttributeError:
            self.__stiffness_matrix = self._stiffness_matrix()
            return self.__stiffness_matrix

    # }}}


#vim: foldmethod=marker
