#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import datetime
import numpy as np
import sympy as syp
import scipy.linalg as LA
import scipy.sparse.linalg as LAS
from pylops import LinearOperator

from colloc.chebOp import ChebOp
from fun import zeros


class GeneralizedEigenvalueOp(LinearOperator):
    r"""
        Implements the right hand side matrix to allow computation of eigenvalues of
        differential operators discretized using ultraS. This is required since we both
        have constraints in the discretized matrix + a change of basis.
    """
    def __init__(self, colloc, dtype=None, *args, **kwargs):
        self.colloc = colloc

        self.shape = self.colloc.S0.shape
        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        """ Implements the action of the following operator:

            /               \
            |               |
            \               /

            This function implements the action of the matrix B of the right hand side
            of the generalized eigenvalue problem.

            A x = λ B x

        """
        arr = self.colloc.projection.dot(x)
        if len(self.colloc.constraints) > 0:
            return np.hstack((np.zeros(len(self.colloc.constraints)), arr))[self.colloc.idenRows]
        else:  # No constraints!
            return arr

    def _rmatvec(self, x):
        assert False, 'Not implemented!'


def eigs(op, f=None, tol=1e-12, n_max=2048, ignore_exception=True,
         sort='lhp', *args, **kwargs):
    # see if operator has tol set
    tol = op.eps
    if f is not None:
        op.setDisc(f.shape[0])
    else:
        # If f is None we will assume that the operator is linear!
        f = zeros(op.neqn, domain=op.domain)

    # Use a QZ decomposition to compute the generalized eigenvalues
    while True:
        try:
            # Get the required matrices!
            #print('f = ', f.shape)
            colloc = op.discretize(f, transform=False, *args, **kwargs)
            M, P, S = colloc.matrix()
            B = GeneralizedEigenvalueOp(colloc)

            Md = M.todense()
            Bd = B.todense()

            AA, BB, alpha, beta, Q, Z = LA.ordqz(Md, Bd, sort=sort,
                                                 overwrite_a=True, overwrite_b=True)

        except ValueError as e:
            if ignore_exception:
                print(e)
                op.n_disc = 2 * op.n_disc
            else:
                raise e

            if op.n_disc >= n_max:
                raise RuntimeError('Error requested discretization is larger than allowed!')
                break

        else:
            break

    mask = np.where(np.hypot(np.real(beta), np.imag(beta)) > tol)
    return alpha[mask] / beta[mask]
