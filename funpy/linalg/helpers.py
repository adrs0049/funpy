#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
from math import ceil
import numpy as np
import scipy as sp
from math import sqrt, log
from scipy.sparse.linalg import splu


def minimumSwaps(arr):
    """
    Minimum number of swaps needed to order a permutation array
    """
    # from https://www.thepoorcoder.com/hackerrank-minimum-swaps-2-solution/
    a = dict(enumerate(arr))
    b = {v:k for k,v in a.items()}
    count = 0
    for i in a:
        x = a[i]
        if x!=i:
            y = b[i]
            a[y] = x
            b[x] = y
            count+=1
    return count


# TODO: move these functions somewhere more appropriate!!
def permutation_vector_to_matrix(E):
    '''Convert a permutation vector E (list or rank-1 array, length n) to a permutation matrix (n by n).
The result is returned as a scipy.sparse.coo_matrix, where the entries at (E[k], k) are 1.
'''
    n = len(E)
    j = np.arange(n)
    return sp.sparse.coo_matrix((np.ones(n), (E, j)), shape=(n, n))


def logdet_detail_superlu(lu):
    diagL = lu.L.diagonal().astype(np.complex128)
    diagU = lu.U.diagonal().astype(np.complex128)

    logdet = np.log(diagL).sum() + np.log(diagU).sum()

    swap_sign = minimumSwaps(lu.perm_r)
    sign = (-1)**(swap_sign) * np.sign(diagL).prod()*np.sign(diagU).prod()

    return np.real(sign), np.real(logdet)


def logdet_detail_umfpack(lu):
    diagL = lu.L.diagonal().astype(np.complex128)
    diagU = lu.U.diagonal().astype(np.complex128)
    diagR = lu.R.astype(np.complex128)

    logdet = np.log(diagL).sum() + np.log(diagU).sum() + np.log(diagR).sum()

    swap_sign = minimumSwaps(lu.perm_r)
    sign = (-1)**(swap_sign) * np.sign(diagL).prod()*np.sign(diagU).prod()

    swap_sign = minimumSwaps(lu.perm_c)
    sign = (-1)**(swap_sign) * np.sign(diagL).prod()*np.sign(diagU).prod()

    return np.real(sign), np.real(logdet)


def logdet(M):
    lu = splu(M.tocsc())
    return logdet_detail_superlu(lu)


def givensAlgorithm(f, g):
    """
    Return a givens Rotation.
    """
    onepar = 1.0
    zeropar = 0.0

    safmn2  = np.finfo(np.double).eps
    safmn2u = np.finfo(np.double).eps
    safmx2  = 1.0 / safmn2
    safmx2u = 1.0 / safmn2

    if g == 0:
        cs = onepar
        sn = zeropar
        r = f
    elif f == 0:
        cs = zeropar
        sn = onepar
        r = g
    else:
        f1 = f
        g1 = g
        scalepar = max(abs(f1), abs(g1))
        if scalepar >= safmx2u:
            count = 0
            while True:
                count += 1
                f1 *= safmn2
                g1 *= safmn2
                scalepar = max(abs(f1), abs(g1))
                if scalepar < safmx2u: break

            r = sqrt(f1 * f1 + g1 * g1)
            cs = f1 / r
            sn = g1 / r
            for i in range(count):
                r *= safmx2

        elif scalepar <= safmn2u:
            count = 0
            while True:
                count += 1
                f1 *= safmx2
                g1 *= safmx2
                scalepar = max(abs(f1), abs(g1))
                if scalepar > safmn2u: break

            r = sqrt(f1 * f1 + g1 * g1)
            cs = f1 / r
            sn = g1 / r
            for i in range(count):
                r *= safmn2

        else:
            r = sqrt(f1 * f1 + g1 * g1)
            cs = f1 / r
            sn = g1 / r

        if abs(f) > abs(g) and cs < 0:
            cs = -cs
            sn = -sn
            r = -r

    return cs, sn, r


def slogdet_hessenberg(mat):
    """
    Computes the sign and logdet of a Hessenberg matrix!
    """
    n, m = mat.shape
    assert n == m, ''

    # TODO: improve me
    mat = mat.astype(np.double)

    # result
    logdet = 0.0
    P = 1.0

    if n == 0:
        return P, logdet

    # temp storage
    g = np.empty(n)
    g[:] = mat[:, m-1]

    for k in range(n-1, 0, -1):
        c, s, r = givensAlgorithm(g[k], mat[k, k-1])
        logdet += log(abs(r))
        P *= np.sign(r)

        g[k-1] = c * (mat[k-1, k-1]) - np.conjugate(s) * g[k-1]
        for j in range(0, k - 1):
            g[j] = c * mat[j, k-1] - np.conjugate(s) * g[j]

    logdet += log(abs(g[0]))
    P *= np.sign(g[0])
    return P, logdet


def detH(A):
    """ Computes the determinant of an upper Hessenberg matrix

        TODO: Implement lower Hessenberg matrix!
    """
    # Upper Hessenberg determinant
    n = A.shape[0]
    p = np.ones(n)
    d = np.ones(n+1)

    # d[0] = 1
    # d[1] = A[0, 0]
    # d[k] =
    #
    d[0] = 1
    d[1] = A[0, 0]

    signs = (-1)**np.arange(1, n, 1)

    # TODO: implement in cython.
    for k in range(1, n):
        # Update the products
        for j in range(k):
            p[j] *= d[j] * A[k, k-1]

        d[k+1] = A[k, k] * d[k] + np.sum(signs[:k][::-1] * A[:k, k] * p[:k])

    return d[k+1]


class Determinant:
    def __init__(self, logdet=None, sign=None, lu=None):
        self.logdet = logdet
        self.sign = sign
        self.lu = lu

    def __call__(self):
        if self.sign is None:
            self.sign, self.logdet = logdet_detail_umfpack(self.lu)
        return self.sign, self.logdet

    def det(self):
        return self.sign * np.exp(self.logdet)
