#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
import scipy.sparse as sps
from pylops import LinearOperator

from ultra import ultra2ultra
from functional import Functional
from newton.lops.NewtonGaussOperatorPrecond import NewtonGaussContinuationCorrectorPrecond
from colloc.tools import replace_bc


class NewtonGaussOp(LinearOperator):
    def __init__(self,
                 matrix, adjoint,
                 da_matrix, da_adjoint,
                 dtype=None, *args, **kwargs):

        self.mat    = matrix
        self.adj    = adjoint
        self.da_mat = da_matrix
        self.da_adj = da_adjoint

        # Basic setup for the linear operator
        self.shape = tuple([sum(x) for x in zip(self.mat.shape, (0, 1))])
        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        """ Implements y = Ax

            Here the matrix A is given by:

                [ D_u F(a, u)      D_a F(a, u) ]

            where D_u F(a, u) is a n x n matrix:
                  D_a F(a, u) is a n x 1 matrix

            Output:  / BC \
                     \ y  /

            where y in R[A] -> need a change of basis here.

        """
        n = self.shape[0]
        y = self.mat @ x[:n] + self.da_mat * x[n]

        # TEMP XXX - doesn't matter right now
        nshape = ((self.shape[0] - 4)//2, 2)
        f = y[4:]
        y[4:] = ultra2ultra(f.reshape(nshape, order='F'), 2, 0).flatten()
        return y

    def _rmatvec(self, y):
        """ Implements x = A^H y

            Here the matrix A is given by:

                / D_u F^H(a, u) \
                \ D_a F^H(a, u) /

            Used for LSMR

            Input here is in BC x C^2

        """
        r = np.dot(self.da_adj, y)
        x = self.adj @ y   # in C2
        x = ultra2ultra(x.reshape((self.shape[0]//2, 2), order='F'), 2, 0).flatten(order='F')
        return np.hstack((x, r))   # Output in (C^0, R)

    def to_matrix(self):
        mat = sps.lil_matrix(self.shape)
        mat[:, :-1] = self.mat
        mat[:, -1]  = self.da_mat.reshape((self.da_mat.size, 1))
        return mat.tocsr()
