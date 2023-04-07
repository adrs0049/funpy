#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
from scipy.special import gamma, gammaln
import scipy.sparse as sps


def jac2jac(c_jac, alpha, beta, gam, delta):
    c_jac, A, B = jacobiIntegerConversion(c_jac, alpha, beta, gam, delta)

    if np.abs(A - gam) > 1e-15:
        assert False, 'Not Supported!'

    if np.abs(B - delta) > 1e-15:
        assert False, 'Not Supported!'

    return c_jac


def jacobiIntegerConversion(c_jac, alpha, beta, gam, delta):

    while alpha <= gam - 1:
        c_jac = RightJacobiConversion(c_jac, alpha, beta)
        alpha += 1

    while alpha >= gam + 1:
        c_jac = LeftJacobiConversion(c_jac, alpha-1, beta)
        alpha -= 1

    while beta <= delta - 1:
        c_jac = UpJacobiConversion(c_jac, alpha, beta)
        beta += 1

    while beta >= delta + 1:
        c_jac = DownJacobiConversion(c_jac, alpha, beta-1)
        beta -= 1

    return c_jac, alpha, beta


def UpJacobiConversion(v, a, b):
    """ Convert Jacobi (alpha, beta) -> (alpha, beta + 1) """
    n, m = v.shape

    d1 = np.hstack((1, (a+b+2)/(a+b+3), np.arange(a+b+3, a+b+n+1) / np.arange(a+b+5, a+b+2*n, 2)))
    d2 = np.arange(a+1, a+n) / np.arange(a+b+3, a+b+2*n, 2)
    D1 = sps.spdiags(d1, 0, n, n)
    D2 = sps.spdiags(d2, 0, n-1, n-1)

    return D1.dot(v) + np.vstack((D2.dot(v[1:, :]), np.zeros((1, m))))


def DownJacobiConversion(v, a, b):
    """ Convert Jacobi (alpha, beta + 1) -> (alpha, beta) """
    n = v.shape[0]

    topRow = np.hstack((1, (a+1)/(a+b+2), (a+1)/(a+b+2) * np.cumprod(np.arange(a+2, a+n) / np.arange(a+b+3,a+b+n+1))))
    topRow = (-1.0)**np.arange(n) * topRow
    tmp = np.multiply(topRow[None, :], v.T)
    vecsum = np.fliplr(np.cumsum(np.fliplr(tmp), 1))
    ratios = np.hstack((1, -(a+b+3)/(a+1), (np.arange(a+b+5, a+b+2*n, 2) / np.arange(a+b+3, a+b+n+1)) * (1./topRow[2:])))
    return np.multiply(ratios, vecsum).T


def RightJacobiConversion(v, a, b):
    """ Convert Jacobi (alpha, beta) -> (alpha+1, beta) """
    v[1::2] = -v[1::2]
    v = UpJacobiConversion(v, b, a)
    v[1::2] = -v[1::2]
    return v


def LeftJacobiConversion(v, a, b):
    """ Convert Jacobi (alpha+1, beta) -> (alpha, beta) """
    v[1::2] = -v[1::2]
    v = DownJacobiConversion(v, b, a)
    v[1::2] = -v[1::2]
    return v


def scl(lam, n):
    if lam == 0:
        nn = np.arange(n)
        s = np.hstack((1, np.cumprod((nn + 0.5)/(nn + 1))))
    else:
        nn = np.arange(n+1)
        s = (gamma(2 * lam) / gamma(lam + 0.5)) * np.exp(gammaln(lam + 0.5 + nn) - gammaln(2 * lam + nn))

    return s


def ultra2ultra(c, lam_in, lam_out):
    n = len(c) - 1
    c /= scl(lam_in, n)[:, None]
    c = jac2jac(c, lam_in - 0.5, lam_in - 0.5, lam_out - 0.5, lam_out - 0.5)
    c *= scl(lam_out, n)[:, None]

    # FIXME!
    if c.shape[0] - 1 != n:
        return c[None, 0, :]

    return c
