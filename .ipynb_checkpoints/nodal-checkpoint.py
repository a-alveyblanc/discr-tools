"""
This file contains everything necessary to set up a simple 1D nodal basis. Also
provides a routine to compute GLL nodes and quadrature weights 
"""
import numpy as np
import numpy.linalg as la

from scipy.interpolate import lagrange
from scipy.special import legendre


class Lagrange:
    """
    Class representing nodal basis functions
    """
    def __init__(self, order, nodes, weights=None):
        self.order = order
        self.nnodes = order + 1
        self.nodes = nodes
        self.weights = weights


    @property
    def basis(self):
        """
        Returns a list of `np.poly1d` Lagrange basis functions as callables
        ```
            Lagrange.basis[0](x)
        ```
        """
        try:
            return self.__basis
        except AttributeError:
            unit_vecs = np.eye(self.nnodes)
            self.__basis = [
                lagrange(self.nodes, unit_vecs[i])
                for i in range(self.nnodes)
            ]
            return self.__basis


    @property
    def basis_dx(self):
        """
        The same as Lagrange.basis but for the derivatives of the basis
        """
        try:
            return self.__basis_dx
        except AttributeError:
            self.__basis_dx = [
                self.__basis[i].deriv()
                for i in range(self.nnodes)
            ]
            return self.__basis_dx


class GLLNodalBasis(Lagrange):
    """
    Creates a nodal basis using GLL nodes
    """
    def __init__(self, order):
        nodes, wts = gll_nodes_weights(order+1)
        super().__init__(order, nodes, wts)


class EquispacedNodalBasis(Lagrange):
    """
    Creates a nodal basis using equispaced nodes
    """
    def __init__(self, order):
        nodes, wts = equispaced_nodes_weights(order+1)
        super().__init__(order, nodes, wts)


def undetermined_coeffs(nodes):
    """
    Method of undetermined coefficients for determining weights when we have
    access to the nodes
    """
    # compute weights via method of undetermined coefficients
    vdm = np.array([
        nodes**i
        for i in range(len(nodes))
    ])

    rhs = np.array([
        1/(j+1) * (1 - (-1)**(j+1))
        for j in range(len(nodes))
    ])

    return la.solve(vdm, rhs)


def equispaced_nodes_weights(n):
    nodes = np.linspace(-1, 1, num=n)
    weights = undetermined_coeffs(nodes) 

    return nodes, weights


def gll_nodes_weights(n):
    """
    Computes a 1D set of GLL quadrature nodes and weights. Nodes are taken to
    be -1 and 1 (the endpoints of the reference domain) and the roots of the 
    derivative of the n-1 Legendre polynomial.

    Method of undetermined coefficients is used to compute the weights.
    """
    # find roots of n-1 Legendre polynomial
    nodes = np.zeros(n)
    nodes[[0,-1]] = -1, 1
    nodes[1:-1] = legendre(n-1).deriv().roots

    weights = undetermined_coeffs(nodes) 
    
    return nodes, weights


#vim: foldmethod=marker
