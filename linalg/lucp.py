#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen

import numpy as np
import numbers
from scipy.sparse import csr_matrix
from scipy.sparse import tril, triu, eye


""" Compute the LU decomposition of a matrix A with full pivoting.
    Thus leading:
                    LU = P A Q

    where:
            L is lower triangular
            U is upper triangular
            P, Q are permutation matrices
"""

def lucp(A, tol=1e-8, full=False, format='csr'):
    n, m = A.shape

    if np.any(np.iscomplex(A)):
        dtype = A.dtype
    elif issubclass(A.dtype.type, numbers.Integral):
        # make sure the matrix is float
        dtype = np.float64
        A = A.astype(dtype)
    else:
        dtype = A.dtype

    # pivot vectors
    p = np.arange(0, n, 1, dtype=np.int64)
    q = np.arange(0, m, 1, dtype=np.int64)

    # temp storage
    rt = np.zeros(m, dtype=dtype)
    ct = np.zeros(n, dtype=dtype)
    t  = np.zeros(1, dtype=np.int64)

    for k in np.arange(0, min(n - 1, m), 1):
        # take abs value
        B = np.abs(A)

        # determine the pivot
        cv, ri = B[k:n, k:m].max(axis=0, keepdims=False), B[k:n, k:m].argmax(0)
        ci     = cv.argmax(0)
        rp     = ri[ci] + k
        cp     = ci + k

        # swap row
        t[:]     = p[k]
        p[k]     = p[rp]
        p[rp]    = t

        rt[:]    = A[k, :]
        A[k, :]  = A[rp, :]
        A[rp, :] = rt

        # swap col
        t[:]     = q[k]
        q[k]     = q[cp]
        q[cp]    = t

        ct[:]    = A[:, k]
        A[:, k]  = A[:, cp]
        A[:, cp] = ct

        if np.abs(A[k, k]) >= tol:
            rows = np.arange(k + 1, n, 1)
            cols = np.arange(k + 1, m, 1)
            # replace everything in the column below the pivot
            A[rows, k] = A[rows, k] / A[k, k]
            idx = np.ix_(rows, cols)
            A[idx] = A[idx] - np.outer(A[rows, k], A[k, cols])
        else:
            # pivot is too small
            break

    l1 = min(n, m)

    if full:
        L = np.tril(A[:n, :l1], -1) + np.eye(n, l1)
        U = np.triu(A[:l1, :])
    else:
        # these matrices are sparse!
        L = tril(A[:n, :l1], -1, format=format) + eye(n, l1, format=format)
        U = triu(A[:l1, :], format=format)

    # generate sparse matrices for the permutation matrices

    if full:
        P = csr_matrix((np.ones_like(p), (np.arange(0, n, 1), p)), shape=(n, n))
        Q = csr_matrix((np.ones_like(q), (q, np.arange(0, m, 1))), shape=(m, m))

        P = P.toarray()
        Q = Q.toarray()

    return L, U, p, q


# A simple test
if __name__ == '__main__':
    A = np.arange(1, 10, 1).reshape((3,3))
    L, U, P, Q = lucp(A, full=True)

    # check that LU = PAQ
    LU = np.dot(L, U)
    Ap = np.dot(P, np.dot(A, Q))
    print('LU:', LU, ' Ap:', Ap)
    print('|diff|:', np.linalg.norm(LU-Ap, ord=2))

    assert np.allclose(LU, Ap, rtol=1e-7, atol=1e-8), 'lucp test failed!'


    # Now do a simple test with sparse matrices
    A = np.arange(1, 10, 1).reshape((3,3))
    L, U, P, Q = lucp(A, full=False)

    # check that LU = PAQ -> need to transform back to a dense layout
    L = L.toarray()
    U = U.toarray()
    P = P.toarray()
    Q = Q.toarray()
    LU = np.dot(L, U)
    Ap = np.dot(P, np.dot(A, Q))

    assert np.allclose(LU, Ap, rtol=1e-7, atol=1e-8), 'sparse-lucp test failed!'

    print('Simple tests passed.')
