# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
#cython: language_level=3
#cython: annotate=True
#cython: infer_types=True
import numpy as np

cimport numpy as np
cimport cython

from numbers import Number
from scipy.fft import ifft, fft

from funpy.cheb.detail import polyfit, polyval
from funpy.cheb.chebpy import chebtec
from funpy.trig.trigtech import trigtech
from funpy.cheb.detail import prolong, simplify_coeffs


def negative(x, **kwargs):
    # Create the new function
    if not isinstance(x, chebtec):
        return np.negative(x, **kwargs)
    else:
        return chebtec(coeffs=-1 * x.coeffs, simplify=False, ishappy=x.ishappy)

def positive(x, **kwargs):
    return x

def absolute(x, **kwargs):
    #abs_values = np.abs(x.get_values())
    # TODO: this needs proper improvement
    #return chebtec(coeffs=polyfit(abs_values), maxLength=2**10,
    #               simplify=False, ishappy=False)  # this will never be happy!
    return chebtec(op=lambda y: np.abs(x(y)), minSamples=x.n, maxLength=2**10)

def power(x1, x2, **kwargs):
    # if we have one element that is a ndarray or Number we want that to be x2!
    if isinstance(x1, Number):
        return NotImplemented

    if isinstance(x2, Number):
        # Do nothing just return the function
        if x2 == 1.0:
            return x1
        return chebtec(op=lambda x: np.power(x1(x), x2), minSamples=2 * x1.n)
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

    # If one of the arguments is a trigtech -> build a chebtec and then call multiply
    elif isinstance(x1, trigtech):
        x1_cheb = chebtec(op=lambda x: x1.feval(x))
        result = add(x1_cheb, x2, **kwargs)
        return trigtech(op=lambda x: result.feval(x))

    elif isinstance(x2, trigtech):
        x2_cheb = chebtec(op=lambda x: x2.feval(x))
        result = add(x1, x2_cheb, **kwargs)
        return trigtech(op=lambda x: result.feval(x))

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
                oc = prolong(oc, nf)
            elif nf < no:
                c = prolong(c, no)

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

    else:
        return NotImplemented

    # Create the new function
    if out is not None:
        out[0].ishappy = ishappy
        return out[0]
    else:
        if isinstance(x1, trigtech) or isinstance(x2, trigtech):
            return trigtech(coeffs=c, simplify=False, ishappy=ishappy)
        else:
            return chebtec(coeffs=c, simplify=False, ishappy=ishappy)

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
    return hc[:mn, :]

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
    elif np.all(f.conj() == g):
        coeffs = coeff_times(f.conj(), g)
        pos = True
    else:
        coeffs = coeff_times(f, g)

    # TODO: check this application of real!
    return np.asfortranarray(np.real(coeffs)), pos

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

    # If one of the arguments is a trigtech -> build a chebtec and then call multiply
    elif isinstance(x1, trigtech):
        x1_cheb = chebtec(op=lambda x: x1.feval(x))
        result = multiply(x1_cheb, x2, **kwargs)
        return trigtech(op=lambda x: result.feval(x))

    elif isinstance(x2, trigtech):
        x2_cheb = chebtec(op=lambda x: x2.feval(x))
        result = multiply(x1, x2_cheb, **kwargs)
        return trigtech(op=lambda x: result.feval(x))

    # We interpret multiplication with another numpy array of size equal to
    # the size of the chebpy object as element-wise multiplication in
    # coefficient space.
    elif isinstance(x2, type(x1)):
        # Multiplication with other chebtec
        if x1.n == 1:  # x1 is a constant function
            return multiply(x2, x1.coeffs, **kwargs)
        elif x2.n == 1:  # x2 is a constant function
            return multiply(x1, x2.coeffs, **kwargs)

        if out is not None:
            c = out[0].coeffs

        # do multiplication in coefficient space
        c, pos = coeff_times_main(x1.coeffs, x2.coeffs)
        c = simplify_coeffs(c)

        # Simply copy the happy status
        ishappy = x1.ishappy and x2.ishappy

        if pos:
            # We know that the product should be positive. However,
            # simplify may have destroyed that property so we enforce it.
            v = polyval(c)
            c = polyfit(np.abs(v))
            c = np.asarray(c, order='F')

        # Let's make sure that the imaginary part in the above is small!
        # TODO: report this as a warning!
        # assert np.max(np.abs(np.imag(c))) < 1e-8, 'chebtec.__mul__ encountered imaginary part!
        # ||Im(c)|| = %.6g.' % (np.max(np.abs(np.imag(c))))
    else:
        return NotImplemented

    # Create the new function
    if out is not None:
        out[0].ishappy = ishappy
        out[0].coeffs = c
        return out[0]
    else:
        return chebtec(coeffs=c, simplify=False, ishappy=ishappy)

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
        fun = chebtec(op=lambda x: np.divide(x1, x2(x)), minSamples=2 * x2.n)
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
        c /= x2
    elif isinstance(x2, np.ndarray):
        if x1.m == 1:
            c /= x2
        else:
            c /= np.tile(x2, (x1.n, 1))
    elif isinstance(x2, type(x1)):
        fun = chebtec(op=lambda x: np.divide(x1(x), x2(x)), minSamples=x1.n + x2.n)
        if out is not None:
            out[0].coeffs = fun.coeffs
            return out[0]
        return fun
    else:
        return NotImplemented

    if out is not None:
        return out[0]
    else:
        return chebtec(coeffs=c, ishappy=x1.ishappy)
