#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
import scipy.linalg as LA
from functools import lru_cache
from scipy.sparse import spdiags, csr_matrix, diags

CACHE_SIZE = 25


def find(array):
    """ Returns row, col, v """
    if len(array.shape) == 1:
        row, = np.nonzero(array)
        return row, np.zeros_like(row), array[row]
    else:  # not ideal
        row, col = np.nonzero(array)
        return row, col, array[row, col]


def sptoeplitz_fast(col, format=None):
    """  """
    n = col.size

    if n < 1e2:
        if np.count_nonzero(col) == 1:
            Ic = np.nonzero(col.squeeze())[0]
            if Ic == 0:
                T = spdiags(col[Ic] * np.ones(n), 0, n, n, format=format)
            else:
                # Diagonals are stored row-wise in the first argument!
                T = spdiags(np.vstack((col[Ic] * np.ones(n), col[Ic] * np.ones(n))),
                            np.hstack((-Ic, Ic)), n, n, format=format)
        else:
            T = LA.toeplitz(col, col)
            T = csr_matrix(T).asformat(format)
    else:
        # locate non-zero diagonals
        ic, jc, sc = find(col)

        # use spdiags for construction
        d = np.hstack((ic[1:], -ic))
        B = np.tile(np.hstack((sc[1:], sc)), (n, 1)).T
        T = spdiags(B, d, n, n, format=format)

    return T


def sptoeplitz(col, row, format=None):
    """  """
    m = col.size
    n = row.size

    if m < 1e2 and n < 1e2:
        if np.count_nonzero(col) == 1 and np.count_nonzero(row) == 1:
            Ic = np.nonzero(col.squeeze())[0]
            Ir = np.nonzero(row.squeeze())[0]
            if Ic == 0:
                T = spdiags(col[Ic] * np.ones(m), 0, m, n, format=format)
            else:
                # Diagonals are stored row-wise in the first argument!
                T = spdiags(np.vstack((col[Ic] * np.ones(m), row[Ir] * np.ones(n))),
                            np.hstack((-Ic, Ir)), m, n, format=format)
        else:
            T = LA.toeplitz(col, row)
            T = csr_matrix(T).asformat(format)
    else:
        # locate non-zero diagonals
        ic, jc, sc = find(col)
        row[0] = 0  # not used
        ir, jr, sr = find(row)

        # use spdiags for construction
        d = np.hstack((ir, -ic))
        B = np.tile(np.hstack((sr, sc)), (min(m, n), 1)).T
        T = spdiags(B, d, m, n, format=format)

    return T


def sphankel(r, format=None):
    """ Sparse Hankel operator """
    n = r.size
    r = np.flipud(r)
    ic, jc, sc = find(r)

    B = np.tile(sc, (n, 1)).T
    return spdiags(B, ic, n, n).tocsr()[:, ::-1]


@lru_cache(maxsize=CACHE_SIZE)
def spconvert(n, lam, format=None):
    """ Compute sparse representation for the conversion operator

        returns the matrix that transforms C^{lam} -> C^{lam + 1}
    """
    assert lam >= 0, 'lam must be non-negative!'
    if lam == 0:
        dg = 0.5 * np.ones(n - 2)
        return spdiags(np.hstack((np.array([[1, 0.5], [0, 0]]), np.vstack((dg, -dg)))),
                       [0, 2], n, n, format=format)
    else:
        dg = lam / (lam + np.arange(2, n))
        return spdiags(np.hstack((np.array([[1, lam / (lam + 1)], [0, 0]]), np.vstack((dg, -dg)))),
                       [0, 2], n, n, format=format)


@lru_cache(maxsize=CACHE_SIZE)
def spconvert_inv(n, lam, format=None):
    """ Computes sparse representation of the conversion operator from

        C^{lam + 1} -> C^{lam}
    """
    assert lam >= 0, 'Lambda must be non-negative!'
    nn = 1 + n // 2 if n & 1 else n // 2

    if lam == 0:
        dg = np.hstack((1, 2 * np.ones(n - 1)))
        return diags(nn * [dg], np.arange(0, n, 2))
    else:
        dg = np.hstack((1, (lam + np.arange(1, n, 1)) / lam))
        return diags(nn * [dg], np.arange(0, n, 2))
