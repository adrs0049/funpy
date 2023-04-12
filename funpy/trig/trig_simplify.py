#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np

from ..cheb.detail import standardChopCmplx
from .transform import coeffs2vals, vals2coeffs


def is_real(values, eps=1e-14, vscl=1):
    return (np.max(np.abs(np.imag(values)), axis=0) <= 3 * eps * vscl).squeeze()

def prolong(coeffs, Nout, isReal=None):
    # If Nout < length(self) -> compressed by chopping
    # If Nout > length(self) -> coefficients are padded by zero
    Nin = coeffs.shape[0]

    if Nout == Nin:  # Do nothing
        return

    if np.remainder(Nin, 2) == 0:
        coeffs = np.vstack((0.5 * coeffs[0, :], coeffs[1:, :], 0.5 * coeffs[0, :]))
        Nin += 1

    if Nin == Nout:
        coeffs = coeffs
        values = coeffs2vals(coeffs)
        values[:, isReal] = np.real(values[:, isReal])

    # Pad with zeros
    if Nout > Nin:
        kup = np.ceil((Nout-Nin)/2).astype(int)
        kdown = np.floor((Nout-Nin)/2).astype(int)
        coeffs = np.vstack((np.zeros((kup, coeffs.shape[1])),
                            coeffs,
                            np.zeros((kdown, coeffs.shape[1]))))

        coeffs = coeffs
        values = coeffs2vals(coeffs)
        values[:, isReal] = np.real(values[:, isReal])

    # chop coefficients
    if Nout < Nin:
        kup = np.floor((Nin-Nout)/2).astype(int)
        kdown = np.ceil((Nin-Nout)/2).astype(int)
        coeffs = coeffs[kup:-kdown, :]
        if kup < kdown:
            coeffs[0, :] = 2*coeffs[0, :]

        coeffs = coeffs
        values = coeffs2vals(coeffs)
        values[:, isReal] = np.real(values[:, isReal])

    return coeffs, values


def simplify_coeffs(coeffs, isReal, eps=1e-14):
    # if the current version is to short make sure it will work with chop
    nold = coeffs.shape[0]
    N = max(17, round(nold * 1.25 + 5))
    coeffs, values = prolong(coeffs, N, isReal)

    # after coefficients have been padded with zeros need to introduce some
    # noise to make standardchop happy
    abs_coeffs = np.abs(coeffs[::-1, :])
    n, m = abs_coeffs.shape
    abs_coeffs = vals2coeffs(coeffs2vals(abs_coeffs))

    # recast tol a row vector
    tol = np.max(eps) * np.ones(m)

    isEven = np.remainder(n, 2) == 0
    if isEven:
        abs_coeffs = np.vstack((abs_coeffs[n-1, :],
                                abs_coeffs[n-2:n//2-1:-1, :] + abs_coeffs[:n//2-1, :],
                                abs_coeffs[n//2-1, :]))
    else:
        abs_coeffs = np.vstack((abs_coeffs[n-1:(n+1)//2-1:-1, :] + abs_coeffs[:(n+1)//2-1, :],
                                abs_coeffs[(n+1)//2-1, :]))

    abs_coeffs = np.flipud(abs_coeffs)
    abs_coeffs = np.vstack((abs_coeffs[0, :], np.kron(abs_coeffs[1:, :], np.vstack((1,1)))))

    # loop through columns to compute cutoff
    cutoff = 1
    for k in range(m):
        cutoff = max(cutoff, standardChopCmplx(abs_coeffs[:, k], tol[k]))

    # take the minimum cutoff.
    cutoff = min(cutoff, nold)

    if np.remainder(cutoff, 2) == 0:
        cutoff = cutoff//2 + 1
    else:
        cutoff = (cutoff-1)//2 + 1

    # put the coefficient vector back together
    if isEven:
        coeffs = np.vstack((0.5 * coeffs[0, :], coeffs[1:], 0.5 * coeffs[-1, :]))
        n += 1

    # use cutoff to trim F
    mid = (n+1)//2

    # chop coefficients
    coeffs = coeffs[mid-cutoff:mid+cutoff-1, :]
    values = coeffs2vals(coeffs)
    return values, coeffs
