#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
import scipy.linalg as LA

from funpy.cheb.chebpts import bary_weights

def diffmat(x, k=1):
    """ Compute the k-th order Barycentric differentiation matrix.

        Maps function values at points x to the values of the derivative at
        those points.
    """
    N = x.size

    if N == 0:
        return None
    elif N == 1:
        return np.zeros((0, 0))

    if k == 0:
        return LA.eye(N)

    # get the barycentric weights
    w = bary_weights(N)

    # compute all differences
    x1 = np.expand_dims(x, axis=1)
    x2 = np.expand_dims(x, axis=0)

    # compute all possible differences
    Dx = x1 - x2

    DxRot = np.rot90(Dx, 2)
    idxTo = np.rot90(np.logical_not(np.triu(np.ones(N))))
    Dx[idxTo] = -DxRot[idxTo]

    # fill identity
    np.fill_diagonal(Dx, np.ones(N))
    Dxi = 1./Dx

    w1 = np.expand_dims(w, axis=0)
    w2 = np.expand_dims(w, axis=1)

    Dw  = w1 / w2
    np.fill_diagonal(Dw, np.zeros(N))

    # k = 1
    D = Dw * Dxi

    # negative sum trick
    np.fill_diagonal(D, np.zeros(N))
    np.fill_diagonal(D, -np.sum(D, axis=1))

    if k == 1:
        return D

    # k = 2
    D = 2. * D * (np.vstack(N*(np.diag(D),)).T - Dxi)
    np.fill_diagonal(D, np.zeros(N))
    np.fill_diagonal(D, -np.sum(D, axis=1))

    if k == 2:
        return D

    for n in range(3, k + 1):
        D = n * Dxi * (Dw * np.vstack(N*(np.diag(D),)).T - D)
        np.fill_diagonal(D, np.zeros(N))
        np.fill_diagonal(D, -np.sum(D, axis=1))

    return D
