#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy
import numpy as np
import scipy.linalg.lapack as LAP
import scipy.sparse as sps
import scipy.linalg as LA
import scipy.sparse.linalg as LAS

try:
    from scipy.sparse import csr_array
except ImportError:
    from scipy.sparse import csr_matrix as csr_array

from linalg.qr import qr
from support.tools import minimumSwaps


def safecall(f, name, *args, **kwargs):
    """Call a LAPACK routine, determining lwork automatically and handling
    error return values"""
    lwork = kwargs.get("lwork", None)
    if lwork in (None, -1):
        kwargs['lwork'] = -1
        ret = f(*args, **kwargs)
        kwargs['lwork'] = ret[-2][0].real.astype(numpy.int_)
    ret = f(*args, **kwargs)
    if ret[-1] < 0:
        raise ValueError("illegal value in %dth argument of internal %s"
                         % (-ret[-1], name))
    return ret[:-2]


class QR:
    def __init__(self, A, tol=np.finfo(float).eps, *args, **kwargs):
        """ AP = QR """
        self.shape = A.shape
        self.tol = tol

        # Note if A is sparse these arguments are ignored.
        pivoting = kwargs.pop('pivoting', True)
        mode = kwargs.pop('mode', 'full')

        # Compute QR
        if sps.issparse(A): A = A.todense()
        self.sparse = sps.issparse(A)

        # TEMP FIXME
        # self.A = A

        # if self.sparse:
        #     import sparseqr
        #     self.Q, self.R, self.C, rank = sparseqr.qr(A)
        #     self.Q = self.Q.tocsc()
        #     self.R = self.R.tocsc()
        #     self.P = np.argsort(self.C)

        #     # Check if R is singular
        #     self.is_singular = abs(self.R[-1, -1]) < tol

        #     self.Rbar = self.R[:-1, :-1] if self.is_singular else None
        #     self.Sbar = self.R[:-1, -1].toarray().reshape(self.shape[0] - 1) if self.is_singular else None

        # else:  # Dense scipy QR
        if pivoting:
            # self.Q, self.R, self.C, self.k = qr(A, pivoting=pivoting, mode=mode)
            #self.Q, self.R, self.C, self.k = qr(A, pivoting=pivoting, mode='raw')
            Q, self.R, self.C = LA.qr(A, pivoting=pivoting, mode='raw')
            self.q, self.tau = Q

            m, n = self.shape
            self.k = np.count_nonzero(self.tau[:-1]) if m <= n else np.count_nonzero(self.tau)

            # The opposite transformation
            self.P = np.argsort(self.C)

            self.Rbar = self.R[:-1, :-1]
            self.Sbar = self.R[:-1, -1]

            # Check if R is singular
            self.is_singular = abs(self.R[-1, -1]) < tol

        else:
            self.Q, self.R, self.k = qr(A, pivoting=pivoting, mode=mode)
            self.is_singular = False

            # The opposite transformation
            self.P = np.arange(self.shape[1])

        # Null-vectors
        self.phi = None
        self.psi = None

        # lapack functions
        self.ormqr = LAP.get_lapack_funcs(('ormqr'), (A, ))
        self.trtrs = LAP.get_lapack_funcs(('trtrs'), (A, ))

    def _solve_triangular(self, A, b, *args, **kwargs):
        if self.sparse:
            return LAS.spsolve_triangular(A, b, *args, **kwargs)
        else:
            return LA.solve_triangular(A, b, *args, **kwargs)

    def _null(self):
        """
            Computes the null-space of the matrix A, assuming that it is one-dimensional.
            After QR decomposition the matrix looks like this:

                / R | S \ / ν \
                | ----- | |---| = 0
                \ 0 | 0 / \ μ /

                Then we let μ = -1.

            The left null-vector is computed from the last column of Q since its columns
            form an orthonormal basis of R[A] and if this is deficient N[A^T].
        """
        # Right null-vector
        y1, info = self.trtrs(self.Rbar, self.Sbar, lower=0, trans=0)
        phi = np.hstack((y1, -1))[self.P]

        # Left null-vector
        n = self.shape[1]
        psi = np.zeros(n)
        psi[n-1] = 1.0
        psi, = safecall(self.ormqr, 'ormqr', 'L', 'N', self.q, self.tau, psi)
        psi = psi.conj().squeeze()
        return phi, psi

    def null(self):
        if self.phi is None or self.psi is None:
            self.phi, self.psi = self._null()

        return self.phi, self.psi

    def det_sign(self):
        """ Compute the sign of the determinant of matrix A """
        if self.shape[0] < self.shape[1]:
            # Then we have to drop some columns -> drop column of elementary matrix
            m = minimumSwaps(np.delete(self.P, np.argmax(self.P), 0))
        else:
            m = minimumSwaps(self.P)

        diagR = self.R.diagonal()
        signR = np.sign(diagR).prod()
        return (-1)**(self.k + m) * signR

    def det(self):
        """ Compute the determinant of matrix A """
        if self.shape[0] < self.shape[1]:
            # Then we have to drop some columns -> drop column of elementary matrix
            m = minimumSwaps(np.delete(self.P, np.argmax(self.P), 0))
        else:
            m = minimumSwaps(self.P)

        diagR = self.R.diagonal()
        signR = np.sign(diagR).prod()
        logdet = np.log(np.abs(diagR)).sum()
        sign = (-1)**(self.k + m) * signR

        return sign * np.exp(np.longdouble(logdet))

    def particular(self, b):
        """
            Assuming that the null-space is one-dimensional
            Now the matrix looks like this.

            / R | S \ / ν \       / y \   / f \
            | ----- | |---| = Q^T |---| = |---|
            \ 0 | 0 / \ μ /       \ z /   \ g /

            Now g = 0 since right hand side must ∈ R[A]. This means μ is a free
            variable, and we choose it to be μ = 0.
        """
        #b = self.Q.T.dot(b)
        b, = safecall(self.ormqr, 'ormqr', 'L', 'T', self.q, self.tau, b)

        if abs(b[-1]) > self.tol:
            raise RuntimeWarning("The solvability condition |f| = {0:.4g} is not satisfied!".format(abs(b[-1])))

        # x = LA.solve_triangular(self.Rbar, b[:-1])
        x, info = self.trtrs(self.Rbar, b[:-1], lower=0, trans=0)
        return np.hstack((x, 0))[self.P]

    def particular_adj(self, b):
        """
            Assuming that the null-space is one-dimensional
            Now the matrix looks like this.

            / R^T | 0 \ / ν \       / y \   / f \
            | ------- | |---| = P^T |---| = |---|
            \ S^T | 0 / \ μ /       \ z /   \ g /

            Now we must have that S^T ν = g for system consistency.
        """
        # x = LA.solve_triangular(self.Rbar.T, b[:-1], lower=True)
        b = b[self.C]
        x, info = self.trtrs(self.Rbar, b[:-1], lower=0, trans=1)

        if abs(np.dot(self.Sbar, x) - b[-1]) > self.tol:
            raise RuntimeWarning("The solvability condition is not satisfied!")

        #return self.Q.dot(np.hstack((x, 0)))
        b = np.hstack((x, 0))
        b, = safecall(self.ormqr, 'ormqr', 'L', 'N', self.q, self.tau, b)
        return b

    def solve(self, b):
        """ Solves Ax = b -> RP^T = Q^T b """
        # xhat = self._solve_triangular(self.R, self.Q.T.dot(b), lower=False)
        # return xhat[self.P]

        cq, = safecall(self.ormqr, 'ormqr', 'L', 'T', self.q, self.tau, b)
        xhat, info = self.trtrs(self.R, cq, lower=0, trans=0)
        return xhat[self.P]

    def solve_adj(self, b):
        #if self.sparse:
        #    xhat = LAS.spsolve_triangular(self.R.T, b[self.C], lower=True)
        #else:
        #    xhat = LA.solve_triangular(self.R, b[self.C], trans='T')

        xhat, info = self.trtrs(self.R, b[self.C], lower=0, trans=1)
        cq, = safecall(self.ormqr, 'ormqr', 'L', 'N', self.q, self.tau, xhat)
        return cq
