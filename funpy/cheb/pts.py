#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
import warnings
from scipy.fft import ifft
from functools import lru_cache


""" Barycentric weights """
@lru_cache(maxsize=25)
def bary_weights(N):
    c = np.hstack((np.ones(N-1), 0.5))
    c[-2::-2] = -1
    c[0] *= 0.5
    return c


""" Quadrature weights for Curtis-Clenshaw """
@lru_cache(maxsize=25)
def quadwts(N):
    if N == 0:
        return np.empty()
    elif N == 1:
        return np.array([2.])

    c = 2./np.hstack((1, 1 - np.arange(2, N, 2)**2))
    c = np.hstack((c, c[1:np.floor(N/2).astype(int)][::-1]))
    w = ifft(c)
    w[0] = 0.5 * w[0]
    w = np.hstack((w, w[0]))
    return w.real


""" Scale Chebyshev nodes to a domain [a, b] """
def scaleNodes(x, interval):
    a = interval[0]
    b = interval[1]

    if a == -1 and b == 1:
        return x

    return 0.5 * b * (1. + x) + 0.5 * a * (1. - x)


""" Scale weights to a domain [a, b] """
def scaleWeights(w, interval):
    a = interval[0]
    b = interval[1]

    if a == -1 and b == 1:
        return w

    return 0.5 * np.diff(interval) * w


""" Barycentric weights for re-sampling for second kind points """
def barymat_impl(y, x, w):
    # need to extend dimensions of both arrays
    y = np.expand_dims(y, axis=1)
    x = np.expand_dims(x, axis=0)

    P = y - x  # All y(j) - x(k)
    P = np.expand_dims(w, axis=0) / P  # All w(k) / (y(j) - x(k))
    P = P / np.expand_dims(np.sum(P, axis=1), axis=1)  # Normalization
    P[np.isnan(P)] = 1.
    return P


def barymat(y, x, w):
    # don't warn in this function since we deal with those warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'divide by zero')
        warnings.filterwarnings('ignore', r'invalid value encountered')
        return barymat_impl(y, x, w)


@lru_cache(maxsize=25)
def chebpts_type1_compute(N):
    if N == 0:
        return []
    elif N == 1:
        return [0]
    else:
        return np.sin(np.pi * np.arange(-N+1, N, 2) / (2. * N))


@lru_cache(maxsize=25)
def chebpts_type2_compute(N):
    if N == 0:
        return []
    elif N == 1:
        return [0]
    else:
        return np.sin(np.pi * np.arange(-N+1, N, 2) / (2. * (N - 1)))


""" Functions to compute everything associated with type-1 points """
def chebpts_type1(N, interval=None):
    # some special cases
    if N == 0:
        x = []
        w = []
        v = []
        t = []
    elif N == 1:
        x = [0]
        w = [2]
        v = [1]
        t = [0.5 * np.pi]
    else: # general case
        x = chebpts_type1_compute(N)

        # FIXME!
        if interval is not None:
            x = scaleNodes(x, interval)

        # quadrature weights
        w = quadwts(N)
        if interval is not None:
            w = scaleWeights(w, interval)

        # barycentric weights
        v = bary_weights(N)

        # angles
        t = np.pi * np.flip(np.arange(0, N, 1)) / (N - 1)

    return np.asarray(x), np.asarray(w), np.asarray(v), np.asarray(t)


""" Functions to compute everything associated with type-2 points """
def chebpts_type2(N, interval=None):
    # some special cases
    if N == 0:
        x = []
        w = []
        v = []
        t = []
    elif N == 1:
        x = [0]
        w = [2]
        v = [1]
        t = [0.5 * np.pi]
    else: # general case
        x = chebpts_type2_compute(N)

        if interval is not None:
            x = scaleNodes(x, interval)

        # quadrature weights
        w = quadwts(N)
        if interval is not None:
            w = scaleWeights(w, interval)

        # barycentric weights
        v = bary_weights(N)

        # angles
        t = np.pi * np.flip(np.arange(0, N, 1)) / (N - 1)

    return np.asarray(x), np.asarray(w), np.asarray(v), np.asarray(t)


def chebpts(N, interval=[-1,1], type=2):
    if type == 1:
        # deal with type-1 points
        x, w, v, t = chebpts_type1(N, interval=interval)
    elif type == 2:
        # deal with type-2 points
        x, w, v, t = chebpts_type2(N, interval=interval)
    else:
        assert False, 'Requested non existing Chebyshev points!'

    return x, w, v, t
