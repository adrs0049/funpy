#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
import scipy.sparse as sps
from functools import lru_cache
from scipy.sparse import spdiags, eye, csr_matrix, lil_matrix, issparse
from sparse.csr import delete_rows_csr, eliminate_zeros_csr

from colloc.ultraS.transform import sptoeplitz, spconvert, spconvert_inv, sphankel
from cheb.chebpts import chebpts_type2, barymat, quadwts
from cheb.detail import standardChop

CACHE_SIZE = 25

@lru_cache(maxsize=CACHE_SIZE)
def diffmat(n, m=1, format=None):
    """ Differentiation matrices for ultraspherical spectral method """
    if m > 0:
        D = spdiags(np.arange(n), 1, n, n, format=format)
        for s in range(1, m):
            D = spdiags(2*s*np.ones(n), 1, n, n, format=format) * D
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
    for s in range(K1, K2+1):
        S = spconvert(n, s, format=format) * S

    # Make sure matrix is returned in CSR format!
    return S

@lru_cache(maxsize=CACHE_SIZE)
def convertmat_inv(n, K1, K2, format=None):
    """ Conversion matrix from C^{K2 + 1} -> C^{K1} """
    S = eye(n, format=format)
    for s in range(K2, K1-1, -1):
        S = spconvert_inv(n, s, format=format) * S
    return S

def multmat(k, f, lam, chop=False, format=None, eps=np.finfo(float).eps):
    """ Multiplication matrices for ultraS

        This function forms the k x k multiplication matrix representing the multiplication of F in the C^(lambda) basis.
    """
    c = f.coeffs
    n = c.shape[0]

    if n == 1:
        # the function f is a scalar
        return c.item() * eye(k, format=format)

    # try to chop this
    if chop:
        cutoff = max(1, standardChop(c, eps, minimum=2))
        if cutoff < n:
            c = c[:cutoff+1]
        n = c.shape[0]

    if n < k:
        c = np.vstack((c, np.zeros((k - n, 1))))
    else:
        c = c[:k]

    if lam == 0:  # multiplication in Chebyshev T coefficients
        c = c / 2
        v = np.vstack((2 * c[0], c[1:]))
        M = sptoeplitz(v, v, format='csr')
        H = lil_matrix((k, k))
        H[1:, :-1] = sphankel(c[1:], format='lil')
        M += H.tocsr()
    elif lam == 1:  # Want the C^{lam} C^{lam} Cheb multiplication matrix
        v = np.vstack((2 * c[0], c[1:]))
        M = sptoeplitz(v, v, format='csr')
        M /= 2
        H = lil_matrix((k, k))
        H[:-2, :-2] = sphankel(c[2:]/2, format='lil')
        M -= H.tocsr()
    else:
        # Convert ChebT to a ChebC^{lam}
        c = convertmat(k, 0, lam - 1, format='dia') * c
        m = 2 * k
        M0 = eye(m, format='dia')
        d1 = np.hstack((1, np.arange(2*lam, 2*lam+m-1))) / np.hstack((1, 2*np.arange(lam+1, lam+m)))
        d2 = (1 + np.arange(m)) / (2*np.arange(lam, lam+m))
        Mx = spdiags(d2, -1, m, m, format='dia') + spdiags(d1, 1, m, m, format='dia')
        M1 = 2*lam*Mx

        # Construct multiplication operator by a three-term recurrence
        c = c.squeeze()
        M = c[0] * M0
        M = M + c[1] * M1
        for nn in range(c.size-2):
            M2 = 2 * (1 + nn + lam)/(nn + 2) * Mx * M1 - (1 + nn + 2 * lam - 1) / (nn + 2) * M0
            M = M + c[nn+2] * M2
            M0 = M1
            M1 = M2
            if np.all(np.abs(c[nn + 3:]) < eps):
                break

        # Extract the sub-matrix
        M = M[:n, :n]

    M = eliminate_zeros_csr(M.tocsr())
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
    I = lil_matrix((k, k))
    I[0, :] = w
    return I.asformat(format)

def realmat(k, format=None):
    """ Projection matrices for ultraS
    """
    I = lil_matrix((k, k))
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
    #    print('normal')
    #    return np.bmat(matrix, **kwargs)
