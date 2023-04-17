#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
from scipy.fft import ifft, fft, ifftshift, fftshift


# TODO: figure out why calling abs directly causes endless invalid floating point errors?
# Internally numpy calls hypot to compute the absolute value of a complex number.
# So this should be fine.
def isHerm(values):
    imag = values - np.conj(values[::-1, :])
    return np.max(np.hypot(np.real(imag), np.imag(imag)), axis=0) == 0.0

def isSkew(values):
    imag = values + np.conj(values[::-1, :])
    return np.max(np.hypot(np.real(imag), np.imag(imag)), axis=0) == 0.0

def vals2coeffs(values):
    """ Convert values at N equally spaced points between [-1, 1) to N trigonmetric coefficients

    If N is odd:
          F(x) = C(1)*z^(-(N-1)/2) + C(2)*z^(-(N-1)/2-1) + ... + C(N)*z^((N-1)/2)

    If N is even:
          F(x) = C(1)*z^(-N/2) + C(2)*z^(-N/2+1) + ... + C(N)*z^(N/2-1)

    where z = exp(1j pi x).

    F(x) interpolates the data [V(1) ; ... ; V(N)] at the N equally
    spaced points x_k = -1 + 2*k/N, k=0:N-1.
    """
    n = values.shape[0]

    if n <= 1:
        return np.copy(values, order='F')

    # test for symmetry
    vals = np.vstack((values, values[0, :]))
    is_herm = isHerm(vals)
    is_skew = isSkew(vals)

    # compute coefficients
    coeffs = (1/n) * fftshift(fft(values, axis=0), axes=0)

    # correct if symmetric
    coeffs[:, is_herm] = np.real(coeffs[:, is_herm])
    coeffs[:, is_skew] = 1j * np.imag(coeffs[:, is_skew])

    # These interpolations are defined on the interval [0, 2*pi), but we want them on
    # (-pi, pi). To fix the coefficients for this we just need to assign
    # c_k = (-1)^k c_k, for k = -(N-1)/2:(N-1)/2 for N odd and k = -N/2:N/2-1 for N even
    if np.remainder(n, 2):
        even_odd_fix = np.expand_dims((-1)**np.arange(-(n-1)/2, n/2, 1), axis=1)
    else:
        even_odd_fix = np.expand_dims((-1)**np.arange(-n/2, n/2, 1), axis=1)

    coeffs = even_odd_fix * coeffs
    return np.asfortranarray(coeffs)  # What's the penalty here?

def coeffs2vals(coeffs):
    """ Convert Fourier coefficients to values at N equally spaced points between [-1, 1],
        where N is the number of coefficients.
    """
    n = coeffs.shape[0]

    if n <= 1:
        return np.copy(coeffs, order='F')

    if np.remainder(n, 2):
        even_odd_fix = np.expand_dims((-1)**np.arange(-(n-1)/2, n/2, 1), axis=1)
    else:
        even_odd_fix = np.expand_dims((-1)**np.arange(-n/2, n/2, 1), axis=1)

    coeffs = even_odd_fix * coeffs

    # test for symmetry
    is_herm = np.max(np.abs(np.imag(coeffs)), axis=0) == 0.0
    is_skew = np.max(np.abs(np.real(coeffs)), axis=0) == 0.0

    # shift the coefficients properly
    values = ifft(ifftshift(n * coeffs, axes=0), axis=0)

    # correct if symmetric
    vals = np.vstack((values, values[0, :]))
    hermvals = (vals + np.flipud(np.conj(vals)))/2
    skewvals = (vals - np.flipud(np.conj(vals)))/2
    values[:, is_herm] = hermvals[:-1, is_herm]
    values[:, is_skew] = skewvals[:-1, is_skew]
    return np.asfortranarray(values)  # What's the penalty here?
