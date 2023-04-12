#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
from scipy.fft import ifft, fft

from .trigtech import trigtech
from .transform import vals2coeffs, coeffs2vals


def circshift(f, a):
    """ Shifts a trigtech by A where A is a real number. """
    n, m = f.shape

    # a must be a scalar!
    # TODO CHECK THIS!
    # TODO generalize this function for periodic functions defined on arbitrary intervals
    #   -> ideally through the Fun interface!

    # The trivial case
    if a == 0:
        return f

    # These interpolations are defined on the interval [0, 2*pi), but we want them on
    # (-pi, pi). To fix the coefficients for this we just need to assign
    # c_k = (-1)^k c_k, for k = -(N-1)/2:(N-1)/2 for N odd and k = -N/2:N/2-1 for N even
    #
    # So a shift by a only requires multiplying the k-th coefficients by (exp(-1i pi a))^k.
    #
    if np.remainder(n, 2):
        even_odd_fix = np.exp(-1j*np.pi*a)**(np.arange(-(n-1)/2, n/2, 1))
    else:
        even_odd_fix = np.exp(-1j*np.pi*a)**(np.arange(-n/2, n/2, 1))

    ncoeffs = f.coeffs * np.expand_dims(even_odd_fix, axis=1)
    nvalues = coeffs2vals(ncoeffs)

    if not np.isreal(a):
        is_real = np.zeros(ncoeffs.shape[1]).astype(bool)
    else:
        is_real = f.isReal

    nvalues[:, is_real] = np.real(nvalues[:, is_real])
    return trigtech(coeffs=ncoeffs, values=nvalues,
                    simplify=False, ishappy=True, isreal=is_real)

def circconv(f, g, domain):
    """ Circular convolution of trigtech objects.

    Only supports smooth periodic functions on [-π, π].
    """
    # No support for array valued trigtechs yet
    if f.m > 1 or g.m > 1:
        return NotImplemented

    # Get the sizes of the trigtech objects
    nf = f.n
    ng = g.n

    # Make sure the trigtech objects have the same underlying size
    if nf > ng:
        g.prolong(nf)
    elif nf < ng:
        f.prolong(ng)
    n = f.n

    # Convolution is just multiplication of the Fourier coefficients.
    # Shift g horizontally to -1
    g = np.roll(g, -1)
    nvalues = 2./n * ifft(fft(f.values, axis=0) * fft(g.values, axis=0), axis=0)
    ncoeffs = vals2coeffs(nvalues)

    # Check the happy status of everything
    ishappy = f.ishappy and g.ishappy
    is_real = f.isReal and g.isReal
    nvalues[:, is_real] = np.real(nvalues[:, is_real])

    # We simplify upon construction of the new function object
    return trigtech(coeffs=ncoeffs, values=nvalues,
                    simplify=True, ishappy=ishappy, isreal=is_real)

def trig_adhesion(f, g, domain):
    """ Non-local adhesion operator """
    # No support for array valued trigtechs yet
    if f.m > 1 or g.m > 1:
        return NotImplemented

    # Get the sizes of the trigtech objects
    nf = f.n
    ng = g.n

    # Make sure the trigtech objects have the same underlying size
    if nf > ng:
        g.prolong(nf)
    elif nf < ng:
        f.prolong(ng)

    ncoeffs = f.coeffs * g.coeffs
    nvalues = coeffs2vals(ncoeffs)

    # Check the happy status of everything
    ishappy = f.ishappy and g.ishappy
    is_real = f.isReal and g.isReal
    nvalues[:, is_real] = np.real(nvalues[:, is_real])

    # We simplify upon construction of the new function object
    return trigtech(coeffs=ncoeffs, values=nvalues,
                    simplify=True, ishappy=ishappy, isreal=is_real)
