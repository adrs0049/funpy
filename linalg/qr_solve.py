#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
from math import sqrt
import scipy.sparse as sps
import scipy.linalg as LA
import scipy.sparse.linalg as LAS

from linalg.qr import qr
from support.tools import minimumSwaps, detH, slogdet_hessenberg


class QRCholesky:
    def __init__(self, A, eps=1e-10, *args, **kwargs):
        self.eps = eps
        self.m, self.n = A.shape

        # Compute QR
        sparse = sps.issparse(A)
        if sparse:
            import sparseqr
            Q, R, C, rank = sparseqr.qr(A)
            self.Rtot = R.tocsr()
        else:
            Q, self.Rtot, C, self.k = qr(A, pivoting=True, mode='full')

        # Compute condition number
        self.cond = self.subcondition(self.Rtot)

        # Need the transpose of Q
        self.Q = Q.transpose()

        # This is a column elementary matrix -> argsort so we can use it for vector indices
        self.C = np.argsort(C)

        # get pseudo rank
        self.rank = kwargs.pop('rank', self.pseudo_rank(self.Rtot))
        #print('pseudo_rank = ', self.pseudo_rank(self.Rtot))
        #if self.rank == 0:
        #    self.rank = self.pseudo_rank(self.Rtot)
        #print('rank = ', self.rank, ' / ', max(self.n, self.m), ' ε = ', self.eps)
        #print('internal_par = ', self.internal_embedding_parameter, ' (m, n) = ', self.m, ', ', self.n)

        # The matrix R1 now has the following form if the matrix A is rank deficient.
        #
        #     / J  | j \
        # A = |--------|
        #     \ 0  | 0 /
        #
        # If A is singular prepare the Penrose pseudo-inverse
        #
        if self.is_singular:
            self.R = self.Rtot[:self.rank, :self.rank]
            self.S = self.Rtot[:self.rank, self.rank:]
            if sparse:
                self.jbar = np.asarray(LAS.spsolve_triangular(self.R, self.S.todense(), lower=False))
            else:
                self.jbar = LA.solve_triangular(self.R, self.S)

            # Create the matrix M
            M = np.eye(max(self.m, self.n) - self.rank) + np.dot(self.jbar.T, self.jbar)
            self.L = LA.cholesky(M)
        else:
            self.R = self.Rtot

        # If we determine the matrix to be rank deficient -> solve using pseudoinverse
        # (sparse, singular)
        solver_methods = {(True,  True): self.solve_singular_sparse,
                          (True,  False): self.solve_nonsingular_sparse,
                          (False, True): self.solve_singular_dense,
                          (False, False): self.solve_nonsingular_dense }

        self.solve = solver_methods[(sparse, self.is_singular)]

    @property
    def is_rank_deficient(self):
        return self.eps * self.cond >= 1

    @property
    def is_singular(self):
        return abs(self.n - self.rank) > 0 or abs(self.n - self.m) > 0

    @property
    def rank_deficiency(self):
        return abs(self.n - self.rank)

    @property
    def rank_deficiency_percent(self):
        return self.rank / self.n

    @property
    def internal_embedding_parameter(self):
        return np.argmax(self.C)

    def det(self):
        """
        This function is written for use during continuation!

        This function effectively drops the last columns.
        """
        if self.m < self.n:
            # Then we have to drop some columns -> drop column of elementary matrix
            m = minimumSwaps(np.delete(self.C, np.argmax(self.C), 0))
        else:
            m = minimumSwaps(self.C)

        diagR = self.Rtot.diagonal().astype(np.complex128)
        signR = np.real(np.sign(diagR).prod())
        logdet = np.real(np.log(diagR).sum())
        sign = (-1)**(self.k + m) * signR
        return sign * np.exp(np.longdouble(logdet))

    def detH(self, col=-1):
        """
            Computes the determinant assuming that the last row was dropped in
            the original matrix!
        """
        # Drop the column that used to be the column of the parameter
        to_delete = self.C[-1]
        if self.m < self.n:
            # Then we have to drop some columns -> drop column of elementary matrix
            idx = np.argwhere(self.C == to_delete)
            nC = np.delete(self.C, idx, 0)
            mask = np.where(nC > to_delete)
            nC[mask] -= 1
            m = minimumSwaps(nC)
        else:
            m = minimumSwaps(self.C)

        H = np.delete(self.Rtot, to_delete, 1)
        sign, logdet = slogdet_hessenberg(H)
        return (-1)**(m + self.k) * sign * np.exp(np.longdouble(logdet))

    def tangent_vector(self):
        """ Should this really be here?

            t = P (-w 1) / sqrt(1 + w^T w)
        """
        tangent = np.vstack((-self.jbar, 1)) / sqrt(1. + np.dot(self.jbar.T, self.jbar))
        return tangent[self.C]

    def subcondition(self, R):
        r = R.diagonal()
        rnn = abs(r[-1])
        if rnn <= np.finfo(float).eps:
            return np.inf
        return abs(r[0]) / rnn

    def pseudo_rank(self, R):
        r = np.abs(R.diagonal())
        s = (r[1:] / r[0]) <= r[0] * self.eps
        if np.all(~s):
            return r.size
        return np.argmax(s) + 1

    def solve_nonsingular_dense(self, b):
        """ We want to solve Ax = b when A is non-singular

            The QR decomposition gives us: A = Q R P^T
            Then solving requires that:

            R [ P^T x ] = Q^T b
        """
        c = np.dot(self.Q, b)
        x = LA.solve_triangular(self.R, c)
        return x[self.C]

    def solve_nonsingular_sparse(self, b):
        """ We want to solve Ax = b when A is non-singular

            The QR decomposition gives us: A = Q R P^T
            Then solving requires that:

            R [ P^T x ] = Q^T b
        """
        c = self.Q.dot(b)
        x = LAS.spsolve_triangular(self.R, c, lower=False)
        return x[self.C]

    def solve_singular_dense(self, b):
        """ Want to solve Ax = b via the pseudo inverse -> x = A^+ b

        Scipy returns the decomposition A P = Q R -> A = Q R P^T

        then A^+ = P^T R^+ Q

        Notation for code below:

        P : -> projection on first component
        Q : -> projection on second component

        """
        c = np.dot(self.Q, b)

        Pyk = -LA.solve_triangular(self.R, c[:self.rank])
        Qyk = np.dot(self.jbar.T, Pyk)

        # solve using Cholesky
        tmp = LA.solve_triangular(self.L, Qyk)
        Qxk = LA.solve_triangular(self.L, tmp, trans=1)

        Pxk = Pyk - np.dot(self.jbar, Qxk)
        return -np.hstack((Pxk, Qxk))[self.C]

    def solve_singular_sparse(self, b):
        """ Want to solve Ax = b via the pseudo inverse -> x = A^+ b

        Scipy returns the decomposition A P = Q R -> A = Q R P^T

        then A^+ = P^T R^+ Q

        Notation for code below:

        P : -> projection on first component
        Q : -> projection on second component

        The mathematical details can be found in: "A Modified Newton Method for the
        Solution of Ill-Conditioned Systems of Nonlinear Equations with Applications
        to Multiple Shooting" P. Deuflhard. Numerische Mathematik 22 289-315 (1974)

        """
        c = self.Q.dot(b)

        Pyk = -LAS.spsolve_triangular(self.R, c[:self.rank], lower=False)
        Qyk = np.dot(self.jbar.T, Pyk)

        # solve using Cholesky
        tmp = LA.solve_triangular(self.L, Qyk)
        Qxk = LA.solve_triangular(self.L, tmp, trans=1)

        Pxk = Pyk - np.dot(self.jbar, Qxk)
        return -np.hstack((Pxk, Qxk))[self.C]
