#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
import scipy.sparse as sps
from functools import lru_cache

try:
    from scipy.sparse import csr_array, csc_array, lil_array
except ImportError:
    from scipy.sparse import csr_matrix as csr_array
    from scipy.sparse import csc_matrix as csc_array
    from scipy.sparse import lil_matrix as lil_array

from scipy.sparse import spdiags, eye, issparse
from sparse.csr import delete_rows_csr, eliminate_zeros
from sparse.csc import delete_cols_csc

from colloc.ultraS.transform import sptoeplitz, sptoeplitz_fast, sphankel
from colloc.ultraS.transform import spconvert, spconvert_inv
from cheb.chebpts import chebpts_type2, barymat, quadwts
from cheb.detail import standardChop

CACHE_SIZE = 25


def zeromat(shape, format='csr'):
    if format == 'csr':
        return csr_array(shape)
    elif format == 'csc':
        return csc_array(shape)
    else:
        raise RuntimeError("Unsupported matrix format {0:s}.".format(format))


@lru_cache(maxsize=CACHE_SIZE)
def diffmat(n, m=1, format=None):
    """ Differentiation matrices for ultraspherical spectral method """
    if m > 0:
        D = spdiags(np.arange(n), 1, n, n, format=format)
        for s in range(1, m):
            D = spdiags(2 * s * np.ones(n), 1, n, n, format=format) * D
        return D
    elif m == 0:
        return eye(n, format=format)
    else:
        return NotImplemented


@lru_cache(maxsize=CACHE_SIZE)
def convertmat(n, K1, K2, format=None):
    """ Conversion matrix used in the ultraspherical spectral method.

        Return the N x N matrix realization of the conversion operator
        between two bases of ultraspherical polynomials.

        The matrix maps N coefficients in a C^{K1} basis to N coefficients
        in a C^{K2 + 1} basis.

        If K2 < K1, S is the N x N identity matrix
    """
    S = eye(n, format=format)
    for s in range(K1, K2 + 1):
        S = spconvert(n, s, format=format) * S

    return S


@lru_cache(maxsize=CACHE_SIZE)
def convertmat_inv(n, K1, K2, format=None):
    """ Conversion matrix from C^{K2 + 1} -> C^{K1} """
    S = eye(n, format=format)
    for s in range(K2, K1 - 1, -1):
        S = spconvert_inv(n, s, format=format) * S

    return S


def multmat(k, f, lam, format=None, eps=np.finfo(float).eps):
    """ Multiplication matrices for ultraS polynomials

        This function forms the k x k multiplication matrix representing
        the multiplication of F in the C^(lambda) basis.
    """
    c = f.coeffs
    n = c.shape[0]

    if n == 1:
        # the function f is a scalar
        return c.item() * eye(k, format=format)

    # try to chop this
    if eps > np.finfo(float).eps:
        cutoff = max(1, standardChop(c.flatten(), eps, minimum=2))
        if cutoff < n:
            c = c[:cutoff + 1]
        n = c.shape[0]

    if n < k:
        c = np.vstack((c, np.zeros((k - n, 1))))
    else:
        c = c[:k]
        n = c.shape[0]

    if lam == 0:  # multiplication in Chebyshev T coefficients
        c = c / 2
        v = np.vstack((2 * c[0], c[1:]))
        M = sptoeplitz_fast(v, format='csr')
        H = csr_array((k, k))
        H[1:, :-1] = sphankel(c[1:])
        M += H
    elif lam == 1:  # Want the C^{lam} C^{lam} Cheb multiplication matrix
        v = np.vstack((2 * c[0], c[1:]))
        M = sptoeplitz_fast(v, format='csr')
        M /= 2
        H = csr_array((k, k))
        H[:-2, :-2] = sphankel(c[2:] / 2)
        M -= H
    else:
        # Convert ChebT to a ChebC^{lam}
        c = convertmat(k, 0, lam - 1, format='csr') * c
        m = 2 * k
        M0 = eye(m, format='csr')
        d1 = np.hstack((1, np.arange(2*lam, 2*lam+m-1))) / np.hstack((1, 2*np.arange(lam+1, lam+m)))
        d2 = (1 + np.arange(m)) / (2*np.arange(lam, lam+m))
        Mx = spdiags(d2, -1, m, m, format='csr') + spdiags(d1, 1, m, m, format='csr')
        M1 = 2*lam*Mx

        # Construct multiplication operator by a three-term recurrence
        c = c.squeeze()
        M = c[0] * M0
        M = M + c[1] * M1
        for nn in range(c.size - 2):
            M2 = 2 * (1 + nn + lam)/(nn + 2) * Mx * M1 - (nn + 2 * lam) / (nn + 2) * M0
            M = M + c[nn+2] * M2
            M0 = M1
            M1 = M2
            if np.all(np.abs(c[nn + 3:]) < eps):
                break

        # Extract the sub-matrix
        M = M[:k, :k]

    #M = eliminate_zeros(M.tocsr())
    return M


def intmat(k, format=None):
    """ Integration matrices for ultraS

        This function forms the k x k integration matrix representing the integration of a function
        F expressed using in a C^(0) basis.

        The block structure of this matrix is as follows:

            /   CCW   \
            |    0    |
            \    0    /

        i.e. a matrix having only a first non-zero row, with the first row being filled by the
        Curtis-Clenshaw weights for integration.
    """
    w = np.hstack((2, 0, 2 / (1 - np.arange(2, k)**2)))
    # set all odd terms to zero -> they don't contribute to the integral!
    w[1::2] = 0
    I = lil_array((k, k))
    I[0, :] = w
    return I.asformat(format)


def realmat(k, format=None):
    """ Projection matrices for ultraS
    """
    I = lil_array((k, k))
    I[0, 0] = 1.0
    return I.asformat(format)


def evalfun(n, location):
    """ The evaluation functional """
    # find the collocation points and create an empty functional
    x, _, v, _ = chebpts_type2(n)
    E = np.zeros((1, n))
    E[0, :] = barymat(location, x, v)
    return E


""" Some helper functions that allow easy rectangularization of matrices """
def delete_rows(matrix, rows_to_delete):
    if issparse(matrix):
        return delete_rows_csr(matrix, rows_to_delete)
    else:
        return np.delete(matrix, rows_to_delete, axis=0)


def delete_cols(matrix, cols_to_delete):
    if issparse(matrix):
        return delete_cols_csc(matrix, cols_to_delete)
    else:
        return np.delete(matrix, cols_to_delete, axis=1)


def blockmat(matrix, **kwargs):
    # need to check the blocks to see whether they are sparse!
    isVector = False
    for block in matrix.flat:
        if block.shape[1] == 1:
            isVector = True

    # Depending on what type of matrices we find call the appropriate block
    # matrix constructor
    if isVector:
        # if we have column vectors we simply need to append those
        return np.vstack(matrix.flatten())
    else:
        return sps.bmat(matrix, **kwargs)

    #if isSparse:
    #    return sps.bmat(matrix, **kwargs)
    #else:
    #    return np.bmat(matrix, **kwargs)


def reduceOne(A, S, m: int, n, adjoint=False):
    """
    Reduces the entries of the column cell arrays A and S from sum(N) x sum(N)
    discretizations to sum(N - M) x sum(N) versions (PA and PS, respectively)
    using the block-projection operator P.

    m: The projection order
    n: dim + dimAdjust

    """
    format = 'csc' if adjoint else 'csr'

    # Projection matrix for US remove the last m coeffs
    P = eye(np.sum(n), format=format)
    nn = np.cumsum(np.hstack((0, n)))
    n = np.asarray([n])

    # v are the row indices which are to be removed by projection
    v = np.empty(0)
    v = np.hstack((v, nn[0] + n[0] - np.arange(1, m + 1))).astype(int)
    P = delete_cols_csc(P, v) if adjoint else delete_rows_csr(P, v)

    # project each component of A and S:
    PA = np.copy(A)
    PS = np.copy(S)

    for j in range(PA.size):
        if (P.shape[1] == A[j].shape[0]) or (P.shape[0] == A[j].shape[1]):
            PA[j] = delete_cols(PA[j], v) if adjoint else delete_rows(PA[j], v)
            PS[j] = delete_cols(PS[j], v) if adjoint else delete_rows(PS[j], v)
        else:
            PA[j] = A[j]
            PS[j] = S[j]

    # TODO: can we deal with those expansion in a better way?
    PA = sps.bmat(np.expand_dims(PA, axis=1))
    PS = sps.bmat(np.expand_dims(PS, axis=1))

    # Don't want to project scalars!
    if m == 0 and A[0].shape[1] < np.sum(n):
        P = eye(A.shape[1], format=format)

    return PA, P, PS
