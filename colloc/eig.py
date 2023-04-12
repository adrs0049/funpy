#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
import scipy.linalg as LA
from pylops import LinearOperator

from funpy import zeros


class GeneralizedEigenvalueOp(LinearOperator):
    r"""
        Implements the right hand side matrix to allow computation of eigenvalues of
        differential operators discretized using ultraS. This is required since we both
        have constraints in the discretized matrix + a change of basis.
    """
    def __init__(self, projection, dtype=None, *args, **kwargs):
        self.proj = projection
        self.numConstraints = self.proj.shape[1] - self.proj.shape[0]

        self.shape = (self.proj.shape[1], self.proj.shape[1])
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
        arr = self.proj.dot(x)
        if self.numConstraints:
            return np.hstack((np.zeros(self.numConstraints), arr))
        else:  # No constraints!
            return arr

    def _rmatvec(self, x):
        return NotImplemented


def eigs(op, f=None, ignore_exception=True, sort='lhp', *args, **kwargs):
    # TODO: ideally we would use ARPACK here.
    # see if operator has tol set
    tol = op.eps
    if f is not None:
        op.setDisc(f.shape[0])
    else:
        # If f is None we will assume that the operator is linear!
        f = zeros(op.neqn, domain=op.domain)

    # Use a QZ decomposition to compute the generalized eigenvalues
    # Get the required matrices!
    colloc = op.discretize(f, *args, **kwargs)
    M, _, P = colloc.matrix()
    B = GeneralizedEigenvalueOp(P)

    # Pass to dense matrices
    Md = M.todense()
    Bd = B.todense()

    # Some default options
    overwrite_a = kwargs.pop('overwrite_a', True)
    overwrite_b = kwargs.pop('overwrite_b', True)
    output = kwargs.pop('output', 'complex')

    try:
        AA, BB, alpha, beta, Q, Z = LA.ordqz(Md, Bd, sort=sort, output=output,
                                             overwrite_a=overwrite_a,
                                             overwrite_b=overwrite_b)

        mask = np.where(np.hypot(np.real(beta), np.imag(beta)) > tol)
        return True, (alpha[mask] / beta[mask])

    except ValueError as e:
        print('Eigenvalue computation failed for !')
        return False, np.asarray([0.0])
