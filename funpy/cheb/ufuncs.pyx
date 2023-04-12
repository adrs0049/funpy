# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
#cython: language_level=3
#cython: annotate=True
#cython: infer_types=True
import numpy as np

cimport numpy as np
cimport cython

from numbers import Number
from scipy.fft import ifft, fft, irfft, rfft

from .chebtech import chebtech
from ..trig.trigtech import trigtech

cimport detail


def negative(x, **kwargs):
    # Create the new function
    if not isinstance(x, chebtech):
        return np.negative(x, **kwargs)
    else:
        return chebtech(coeffs=-1 * x.coeffs, simplify=False, eps=x.eps,
                       maxLength=x.maxLength, ishappy=x.ishappy)

def positive(x, **kwargs):
    return x

def absolute(x, **kwargs):
    return chebtech(op=lambda y: np.abs(x(y)), minSamples=x.n,
                   maxLength=2**10, eps=x.eps)

def power(x1, x2, **kwargs):
    # if we have one element that is a ndarray or Number we want that to be x2!
    if isinstance(x1, Number):
        return NotImplemented

    if isinstance(x2, Number):
        # Do nothing just return the function
        if x2 == 1.0:
            return x1
        return chebtech(op=lambda x: np.float_power(x1(x), x2), minSamples=x1.n, eps=x1.eps)
    else:
        return NotImplemented

def add(x1, x2, **kwargs):
    # if we have one element that is a ndarray or Number we want that to be x2!
    if isinstance(x1, (np.ndarray, Number)):
        return add(x2, x1, **kwargs)

    out = kwargs.get('out', None)
    if isinstance(x2, Number) or isinstance(x2, np.ndarray):
        # get the coefficients of x1
        if out is not None:
            c = out[0].coeffs
        else:
            c = np.copy(x1.coeffs, order='F')

        other = np.asanyarray(x2, order='F')

        # the first coefficient represents the constant
        c[0, :] += other.squeeze()

        # simply add to the values
        ishappy = x1.ishappy
        eps = x1.eps

    # If one of the arguments is a trigtech -> build a chebtech and then call multiply
    elif isinstance(x1, trigtech):
        eps = max(x1.eps, x2.eps)
        x1_cheb = chebtech(op=lambda x: x1.feval(x), eps=eps)
        result = add(x1_cheb, x2, **kwargs)
        return trigtech(op=lambda x: result.feval(x), eps=eps)

    elif isinstance(x2, trigtech):
        eps = max(x1.eps, x2.eps)
        x2_cheb = chebtech(op=lambda x: x2.feval(x), eps=eps)
        result = add(x1, x2_cheb, **kwargs)
        return trigtech(op=lambda x: result.feval(x), eps=eps)

    elif isinstance(x2, type(x1)):  # other is expected to be a chebfun object now
        nf = x1.n
        no = x2.n

        if out is not None:
            if nf > no:
                x2.prolong(nf)
            elif nf < no:
                x1.prolong(no)
            c = out[0].coeffs
            oc = x2.coeffs
        else:
            c = np.copy(x1.coeffs, order='F')
            oc = x2.coeffs
            if nf > no:
                oc = detail.prolong(oc, nf)
            elif nf < no:
                c = detail.prolong(c, no)

        # update values and coefficients
        c += oc

        # look for zero output
        absCoeffs = np.abs(c)

        # If absolute coefficients are smaller than tolerance create zero function
        tol = x1.eps * np.max(np.maximum(x1.vscl(), x2.vscl()))
        if np.all(absCoeffs < 0.2 * tol):
            c = np.zeros((1, x1.m), order='F')

        # update values
        ishappy = x1.ishappy and x2.ishappy
        eps = max(x1.eps, x2.eps)

    else:
        return NotImplemented

    # Create the new function
    if out is not None:
        out[0].ishappy = ishappy
        out[0].eps = eps
        return out[0]
    else:
        if isinstance(x1, trigtech) or isinstance(x2, trigtech):
            return trigtech(coeffs=c, simplify=False, ishappy=ishappy, eps=eps)
        else:
            return chebtech(coeffs=c, simplify=False, ishappy=ishappy, eps=eps)

def subtract(x1, x2, **kwargs):
    return add(x1, negative(x2), **kwargs)

""" Multiplication support """
def coeff_times(fc, gc):
    """ Multiplication in coefficient space """
    mn = fc.shape[0]
    t = np.vstack((2 * fc[0, :], fc[1:, :])) # Toeplitz vector
    x = np.vstack((2 * gc[0, :], gc[1:, :])) # Toeplitz vector
    xprime = fft(np.vstack((x, x[:0:-1, :])), axis=0) # FFT for circulant mult
    aprime = fft(np.vstack((t, t[:0:-1, :])), axis=0)
    Tfg = ifft(aprime * xprime, axis=0)
    hc = 0.25 * np.vstack((Tfg[0, :], Tfg[1:, :] + Tfg[:0:-1]))
    return hc[:mn, :].real

def coeff_times_main(f, g):
    # get the sizes
    fn, fm = f.shape
    gn, gm = g.shape

    # prolong: length(f * g) = length(f) + length(g) - 1
    f = np.vstack((f, np.zeros((gn - 1, fm))))
    g = np.vstack((g, np.zeros((fn - 1, gm))))

    # check dimensions

    # There are two cases in which the output is known to be positive, namely:
    # F == conj(G) or F == G and isreal(F)
    pos = False

    # multiply values
    if np.all(f == g):
        coeffs = coeff_times(f, g)
        if np.all(np.isreal(f)):
            pos = True
    elif np.all(np.conj(f) == g):
        coeffs = coeff_times(np.conj(f), g)
        pos = True
    else:
        coeffs = coeff_times(f, g)

    return np.asfortranarray(coeffs), pos

def multiply(x1, x2, **kwargs):
    # if we have one element that is a ndarray or Number we want that to be x2!
    if isinstance(x1, (np.ndarray, Number)):
        return multiply(x2, x1, **kwargs)

    out = kwargs.get('out', None)
    if isinstance(x2, Number):
        # get the coefficients of x1
        if out is not None:
            out[0].coeffs = x1.coeffs * x2
            c = out[0].coeffs
        else:
            c = x1.coeffs * x2

        # simplify zero functions
        if not np.any(c):
            c = np.zeros_like(c, order='F')

        ishappy = x1.ishappy
        eps = x1.eps

    elif isinstance(x2, np.ndarray):
        if x2.size == 1:
            return multiply(x1, x2.item(), **kwargs)

        if len(x2.shape) == 1:
            x2 = np.expand_dims(x2, axis=1)

        # Otherwise the shape of the ndarray must be the same as
        # for the coefficients
        assert x1.shape == x2.shape, 'Shape %s and %s mismatch!' % (x1.shape, x2.shape)

        # multiply the coefficients
        if out is not None:
            # Do this because the various multiply reorders may flip x1 and x2
            out[0].coeffs = x1.coeffs * x2
            c = out[0].coeffs
        else:
            c = x1.coeffs * x2

        ishappy = x1.ishappy
        eps = x1.eps

    # If one of the arguments is a trigtech -> build a chebtech and then call multiply
    elif isinstance(x1, trigtech):
        eps = max(x1.eps, x2.eps)
        x1_cheb = chebtech(op=lambda x: x1.feval(x), eps=eps)
        result = multiply(x1_cheb, x2, **kwargs)
        return trigtech(op=lambda x: result.feval(x), eps=eps)

    elif isinstance(x2, trigtech):
        eps = max(x1.eps, x2.eps)
        x2_cheb = chebtech(op=lambda x: x2.feval(x), eps=eps)
        result = multiply(x1, x2_cheb, **kwargs)
        return trigtech(op=lambda x: result.feval(x), eps=eps)

    # We interpret multiplication with another numpy array of size equal to
    # the size of the chebpy object as element-wise multiplication in
    # coefficient space.
    elif isinstance(x2, type(x1)):
        # Multiplication with other chebtech
        if x1.n == 1:  # x1 is a constant function
            return multiply(x2, x1.coeffs, **kwargs)
        elif x2.n == 1:  # x2 is a constant function
            return multiply(x1, x2.coeffs, **kwargs)

        if out is not None:
            c = out[0].coeffs

        # do multiplication in coefficient space
        eps = max(x1.eps, x2.eps)
        c, pos = coeff_times_main(x1.coeffs, x2.coeffs)
        c = detail.simplify_coeffs(c, eps=eps)

        # Simply copy the happy status
        ishappy = x1.ishappy and x2.ishappy

        if pos:
            # We know that the product should be positive. However,
            # simplify may have destroyed that property so we enforce it.
            v = detail.polyval(c, 1)
            c = detail.polyfit(np.abs(v), 1)
            c = np.asarray(c, order='F')
    else:
        return NotImplemented

    # Create the new function
    if out is not None:
        out[0].ishappy = ishappy
        out[0].coeffs = c
        out[0].eps = eps
        return out[0]
    else:
        return chebtech(coeffs=c, simplify=False, ishappy=ishappy, eps=eps)

def divide(x1, x2, **kwargs):
    return true_divide(x1, x2, **kwargs)

def true_divide(x1, x2, **kwargs):
    """
    Implements division where at least one of x1 or x2 is a chebtech object!
    """
    # Deal with assignment operators
    out = kwargs.pop('out', None)

    # Deal with this first!
    if isinstance(x1, Number):
        eps = x2.eps
        fun = chebtech(op=lambda x: np.divide(x1, x2(x)), minSamples=2 * x2.n, eps=eps)
        if out is not None:
            out[0].coeffs = fun.coeffs
            return out[0]
        return fun
    elif isinstance(x1, np.ndarray):
        if x1.size == 1:
            return true_divide(x1.item(), x2)
        else:
            return NotImplemented

    # We know must have that x1 is a chebtech object.
    if out is not None:
        c = out[0].coeffs
    else:
        c = np.copy(x1.coeffs)

    # Deal with the division!
    if isinstance(x2, Number):
        eps = x1.eps
        c /= x2
    elif isinstance(x2, np.ndarray):
        eps = x1.eps
        if x1.m == 1:
            c /= x2
        else:
            c /= np.tile(x2, (x1.n, 1))
    elif isinstance(x2, type(x1)):
        eps = max(x1.eps, x2.eps)
        fun = chebtech(op=lambda x: np.divide(x1(x), x2(x)),
                      minSamples=x1.n + x2.n, eps=eps)
        if out is not None:
            out[0].coeffs = fun.coeffs
            return out[0]
        return fun
    else:
        return NotImplemented

    if out is not None:
        out[0].eps = eps
        return out[0]
    else:
        return chebtech(coeffs=c, ishappy=x1.ishappy, eps=eps)
