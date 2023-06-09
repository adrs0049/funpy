#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
from math import ceil
import numpy as np

from funpy.cheb import polyval
from funpy.cheb import quadwts


def findNextPowerOf2(n):
    if n <= 1:
        return 2

    n = n - 1
    while n & n - 1:
        n = n & n - 1

    return n << 1


def round_up(num, divisor):
    return ceil(num / divisor) * divisor


def secant(s1, d1, s2, d2):
    """ The function carries out one iteration of the secant method. The iteration
        is given by the equation:

                                s_{n}  -  s_{n-1}
        s_{n + 1} = s_{n} -   ---------------------    q(s_{n - 1})
                             q(s_{n}) - q(s_{n - 1})

        where q(s) is the function whose zero we are seeking.

        Argument:
            s1 -> s_{n - 1}
            s2 -> s_{n}

            d2 -> q(s_{n})
            d1 -> q(s_{n - 1})
    """
    return s2 - d2 * (s2 - s1) / (d2 - d1)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def find_nearest_idx(array, value):
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()


def orientation_y(u):
    """
        Computes the orientation on the function space

        Y = { (u, v) : \int u + v dx = 0 }

        in which an element is defined as:

            positive <=> ∫ u > 0
            negative <=> ∫ u < 0

        Note that the only element for which the integral vanishes is given by (u, v) = (0, 0).
    """
    # print('u:', u)
    if u.shape[0] == 1:
        return np.sign(u[0].coeffs)
    return np.sign(u[0].lval()).squeeze()

    # sum = 0.5 * np.real(np.sum(u[0] - u[1])).squeeze()
    # print('sum = ', sum)
    # if np.abs(sum) <= np.finfo(float).eps:
    #     return 0
    # return np.sign(sum)


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n-1:] / n


def theta_interp(s1, s2, kappa, kc=0.85, wd=1):
    f1 = lambda k: 0.5 * (np.tanh(-(k - kc) / 0.3) + 1)
    f2 = lambda k: 0.5 * (np.tanh( (k - kc) / 0.3) + 1)
    return np.round(f1(kappa) * s1 + f2(kappa) * s2, decimals=3)



def functional(function):
    rescaleFactor = 0.5 * np.diff(function.domain)
    psic = np.empty(function.m, dtype=object)
    w = quadwts(function.n)
    for j, col in enumerate(function):
        psic[j] = np.rot90(polyval(w * col.values.T), -1)
    psic = np.hstack(psic) * rescaleFactor
    return psic


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
