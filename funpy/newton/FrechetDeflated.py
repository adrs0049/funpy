#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
from pylops import LinearOperator
import numpy as np


class FrechetDeflated(LinearOperator):
    r"""
        This generates the operator representing the derivative of a deflated
        Newton's method at the point u.

        So during construction we evaluate the map u -> D_u G[u]

        Given a nonlinear root-finding problem:  F(a, u) = 0
        and a set of known solutions { u_k }_{k=1}^{N}

        the deflated residual is given by:

                       N
            G(a, u) =  π    (sigma + 1 / || u - u_k || )  F(a, u)
                      k = 1

        where the norm is appropriately chosen for the problem at hand.
    """
    def __init__(self, matrix, adjoint=None, par=False, dtype=None, *args, **kwargs):
        """ Arguments:

            u:  Function; location of linearization.
            us: Nonlinear operator source -> carries out linearization
        """
        super().__init__(dtype=np.dtype(dtype), shape=matrix.shape)

        # Local symbols
        self.mat = matrix
        self.adj = adjoint

        # Setup required LinOp information
        self.explicit = False

        # known solutions
        self.eta        = kwargs.pop('eta', 1.0)
        self.b          = kwargs.pop('b', None)
        self.functional = kwargs.pop('deta', None)

    def _matvec(self, x):
        """ Implements y = Ax

            Here this is eta * (A * x - outer(b, deta))
        """
        # If no known solutions we bail and simply compute the regular part
        if self.functional is None:
            return self.mat._mul_vector(x)

        # Compute the derivative of eta
        deta = self.functional(x)

        # The b here should be just F(u)
        return self.eta * self.mat._mul_vector(x) + self.b * deta

    def _rmatvec(self, y):
        assert False, ''
        return self.eta * self.adj._mul_vector(y)

    def to_matrix(self):
        """ Faster than tosparse -> since whenever we are not deflating we
            have know the sparse matrix!
        """
        if self.functional is None:
            return self.mat
        else:
            return self.tosparse()
