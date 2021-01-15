#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np

""" Computes the derivatives of chebyshev polynomials """
def computeDerCoeffs(c):
    """ Recurrence relation for coefficients of derivative. C is the matrix
    of Chebyshev coefficients of a (possible array-valued) chebtec object.
    Cout is the matrix of coefficients for a chebtec object whose columns are
    the derivatives of those of the original.
    """
    if c.ndim == 1:
        c = np.expand_dims(c, axis=1)
    n, m = c.shape
    cout = np.zeros((n-1,m), order='F')
    w = np.tile(np.expand_dims(2*np.arange(1, n), axis=1), (1, m))
    v = w * c[1:, :]
    cout[n-2::-2, :] = np.cumsum(v[n-2::-2, :], axis=0)
    cout[n-3::-2, :] = np.cumsum(v[n-3::-2, :], axis=0)
    cout[0, :]  = 0.5 * cout[0, :]
    return cout

def diff(f, k):
    """ Compute the k-th derivative of the chebtec f """
    c = np.copy(f.coeffs)
    n = c.shape[0]

    # return zero if differentiating too much
    if k >= n:
        return chebtec(coeffs=np.zeros_like(c))

    # Iteratively compute the coefficients of the derivatives
    for m in range(k):
        c = computeDerCoeffs(c)

    # This returns a chebtec that has only N - k Chebyshev coefficients
    return chebtec(coeffs=c)
