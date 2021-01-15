#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
import scipy
import scipy.sparse as sps
import scipy.linalg as LA
import scipy.sparse.linalg as LAS
from sparse.csr import csr_vappend
from pylops import LinearOperator

from states.State import ContinuationState
from support.Functional import Functional
from newton.deflated_residual import DeflatedResidual
from cheb.chebpts import quadwts


class NewtonGaussContinuationCorrectorPrecond(LinearOperator):
    def __init__(self, Du, dtype=None, *args, **kwargs):
        self.Du = Du

        # Basic setup for the linear operator
        self.shape = tuple([sum(x) for x in zip(self.Du.shape, (1, 1))])
        self.dtype = np.dtype(dtype)
        self.explicit = False

        # assemble matrix
        self.mat = sps.lil_matrix(self.shape)
        self.mat[:-1, :-1] = self.Du.to_matrix()
        self.mat[-1, -1] = 1.0
        self.mat = self.mat.tocsr()

    def _matvec(self, x):
        return self.Du._matvec(x[:-1]) + x[-1]

    def to_matrix(self):
        return self.mat


class NewtonGaussContinuationCorrector(LinearOperator):
    """ Implements the linear operator for pseudo-arclength continuation.
    """
    def __init__(self, state, operator, ks=[], dtype=None, *args, **kwargs):
        self.Du = DeflatedResidual(state, operator, par=True, ks=ks, dtype=dtype)
        self.Da = self.Du.colloc.diff_a(state)

        # Basic setup for the linear operator
        self.shape = tuple([sum(x) for x in zip(self.Du.shape, (0, 1))])
        self.dtype = np.dtype(dtype)
        self.explicit = False

        # assemble matrix
        self.mat = sps.lil_matrix(self.shape)
        self.mat[:, :-1] = self.Du.to_matrix()
        self.mat[:, -1] = self.Da.reshape((self.Da.size, 1))
        self.mat = self.mat.tocsr()

        # Also deal with the right hand side
        self.b = self.Du.b

    def rhs(self, state):
        return self.Du.rhs(state)

    def precond(self):
        return NewtonGaussContinuationCorrectorPrecond(self.Du.precond())

    def inverse_basis(self):
        return self.Du.inverse_basis()

    def matrix_full(self, state):
        full_u = self.Du.matrix_full()
        # TODO return full version of this!
        full_a = self.Du.colloc.diff_a(state)
        # TODO: improve this write version of this for CSC
        return csr_vappend(full_u.T.tocsr(), sps.csr_matrix(full_a)).T

    def adjoint(self, state):
        adjoint_u = self.Du.adjoint()
        adjoint_a = self.Du.colloc.diff_a_adjoint(state)
        return csr_vappend(adjoint_u.tocsr(), sps.csr_matrix(adjoint_a))

    def _matvec(self, dx):
        """ Implements y = Ax

            Here the matrix A is given by:

                [ D_u F(a, u)      D_a F(a, u) ]

            where D_u F(a, u) is a n x n matrix:
                  D_a F(a, u) is a n x 1 matrix
        """
        # The action of this matrix can then be computed by
        return self.Du._matvec(dx[:-1]) + self.Da * dx[-1]

    def _rmatvec(self, x):
        """ Implements x = A^H y """
        assert False, ''
        return np.zeros_like(x)

    def to_matrix(self):
        return self.mat
