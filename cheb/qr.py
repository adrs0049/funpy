#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
import scipy.linalg as LA
from funpy.cheb.chebpts import chebpts
from funpy.cheb.detail import polyfit
from funpy.qr import abstractQR

def qr(f, eps=1.48e-8):
    """ Returns the QR factorisation of F such that F = Q * R, where the
        chebtech Q is orthogonal with respect to the L2-norm
    """
    n, m = f.shape

    if m == 0:
        return f, None

    # If it contains only one column we simply scale
    if m == 1:
        R = np.sqrt(f.innerproduct(f))
        Q = f/R
        return Q, R

    tol = eps * np.max(f.vscale)

    # make discrete analog of f
    newN = 2 * max(n, m)
    # get values?
    f.prolong(newN)
    A = f.values

    # Get the chebyshev points
    x, w, _, _ = chebpts(newN)

    # generate discrete E (Legendre-Chebyshev-Vandermonde matrix) directly
    E = np.ones(A.shape)
    E[:, 1] = x
    for k in range(3, m+1):
        E[:, k-1] = ((2 * k - 3) * x * E[:, k-2] - (k - 2) * E[:, k-3]) / (k - 1)

    # Scaling
    for k in range(1, m+1):
        E[:, k-1] = E[:, k-1] * np.sqrt((2 * k - 1) / 2)

    # call abstract QR method
    Q, R = abstractQR(A, E,
                      lambda f, g: np.dot(np.conj(f) * w, g),
                      lambda f: LA.norm(f, ord=np.inf), tol)

    # compute the corresponding Chebyshev coefficients
    coeffs = polyfit(Q)
    return coeffs[:newN//2+1, :], R
    #Q = chebtec(coeffs=coeffs[:newN//2+1, :], interval=f.interval, simplify=False)
    return Q, R
