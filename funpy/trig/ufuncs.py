#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
from numbers import Number

from ..cheb.chebtech import chebtech
import funpy.cheb.ufuncs as cb_funcs # This feels wrong!

from .trig_simplify import prolong, simplify_coeffs
from .trigtech import trigtech
from .transform import coeffs2vals, vals2coeffs


def negative(x, **kwargs):
    # Create the new function
    if not isinstance(x, trigtech):
        return np.negative(x, **kwargs)
    else:
        return trigtech(values=-1 * x.values, coeffs=-1 * x.coeffs,
                        simplify=False, ishappy=x.ishappy)

def positive(x, **kwargs):
    return x

def fabs(x, **kwargs):
    abs_values = np.abs(x.get_values())
    return trigtech(coeffs=vals2coeffs(abs_values), values=abs_values,
                    simplify=False, ishappy=x.ishappy, isreal=x.isReal)

def power(x1, x2, **kwargs):
    # if we have one element that is a ndarray or Number we want that to be x2!
    if isinstance(x1, Number):
        return power(x2, x1)

    if isinstance(x2, Number):
        return trigtech(op=lambda x: np.power(x1(x), x2), minSamples= 2* x1.n)
    else:
        return NotImplemented

def add(x1, x2, **kwargs):
    # if we have one element that is a ndarray or Number we want that to be x2!
    if isinstance(x1, (np.ndarray, Number)):
        return add(x2, x1, **kwargs)

    out = kwargs.pop('out', None)
    if isinstance(x2, Number) or isinstance(x2, np.ndarray):
        # get the coefficients of x1
        if out is not None:
            c = out[0].coeffs
            v = out[0].values
        else:
            c = np.copy(x1.coeffs, order='F')
            v = np.copy(x1.values, order='F')

        other = np.asanyarray(x2, order='F')

        # simply add to the values
        v += other

        # Update coefficients
        if other.shape and other.shape[1] > 1 and x1.m == 1:
            c = np.tile(x1.coeffs, (1, other.shape[1]))

        n = x1.n
        # determine index of constant coefficient term
        if n & 1:
            const_index = (n+1)//2 - 1
        else:
            const_index = n//2

        c[const_index, :] += other

        # update is real
        # TODO FIX THE EPS HERE!
        is_real = x1.isReal & np.all(np.imag(other) < 1e-14)

        # Don't I need to do this?
        ishappy = x1.ishappy

    # If one of the arguments is a chebtech -> map everything to a chebtech and then map back!
    elif isinstance(x1, chebtech) or isinstance(x2, chebtech):
        x1_cheb = chebtech(op=lambda x: x1.feval(x))
        x2_cheb = chebtech(op=lambda x: x2.feval(x))
        result = cb_funcs.add(x1_cheb, x2_cheb)
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
            v = out[0].values
            # print('out!')
            # print('v = ', v.dtype, ' out = ', out[0].values.dtype)

            oc = x2.coeffs
            ov = x2.values
        else:
            # FIXME - I shouldn't need any of these silly type conversions!
            c = np.copy(x1.coeffs, order='F').astype(complex)
            v = np.copy(x1.values, order='F').astype(complex)

            oc = x2.coeffs
            ov = x2.values
            if nf > no:
                oc, ov = prolong(oc, nf, x2.isReal)
            elif nf < no:
                c, v = prolong(c, no, x1.isReal)

            # print('not out!')

        # Get the new values
        # print('out = ', out)
        # if out is not None:
        #     print('out = ', out[0].values.dtype)
        # print('ov=', ov.dtype)
        # print('v=', v.dtype)
        v += ov

        # update is real
        is_real = x1.isReal & x2.isReal

        # Update values that are real
        v[:, is_real] = np.real(v[:, is_real])

        # update the coefficients
        if out is not None:
            out[0].coeffs = vals2coeffs(v)
        else:
            c = vals2coeffs(v)

        # Look for zero output - TODO check that i don't do this before updating c!
        if ~np.any(v) or ~np.any(c):
            # create a zero trigtech
            c = np.zeros((1, v.shape[1]))
            v = np.zeros((1, v.shape[1]))
            ishappy = x1.ishappy
        else:
            ishappy = x1.ishappy and x2.ishappy

    else:
        return NotImplemented

    # Create the new function
    if out is not None:
        out[0].ishappy = ishappy
        out[0].isReal = is_real
        return out[0]
    else:
        return trigtech(values=v, coeffs=c, simplify=False, ishappy=ishappy, isreal=is_real)

def subtract(x1, x2, **kwargs):
    return add(x1, negative(x2), **kwargs)

def multiply(x1, x2, **kwargs):
    # if we have one element that is a ndarray or Number we want that to be x2!
    if isinstance(x1, (np.ndarray, Number)):
        return multiply(x2, x1, **kwargs)

    out = kwargs.pop('out', None)
    if isinstance(x2, Number):
        # get the coefficients of x1
        if out is not None:
            c = out[0].coeffs
            v = out[0].values
        else:
            # FIXME???
            c = np.copy(x1.coeffs, order='F').astype(complex)
            v = np.copy(x1.values, order='F').astype(complex)

        # update coefficients
        c *= x2
        v *= x2

        # update the is real number
        is_real = x1.isreal and np.isreal(x2)
        ishappy = x1.ishappy

    elif isinstance(x2, np.ndarray):
        if x2.size == 1:
            return multiply(x1, x2.item(), **kwargs)

        if len(x2.shape) == 1:
            x2 = np.expand_dims(x2, axis=1)

        # Otherwise the shape of the ndarray must be the same as
        # for the coefficients
        assert x1.shape == x2.shape, 'Shape %s and %s mismatch!' % (x1.shape, x2.shape)

        if out is not None:
            c = out[0].coeffs
            v = out[0].values
        else:
            c = x1.coeffs
            v = x1.values

        # multiply the coefficients
        c *= x2
        v *= x2

        is_real = x1.isreal and np.isreal(x2)
        ishappy = x1.ishappy

    # If one of the arguments is a chebtech -> map everything to a chebtech and then map back!
    elif isinstance(x1, chebtech) or isinstance(x2, chebtech):
        x1_cheb = chebtech(op=lambda x: x1.feval(x))
        x2_cheb = chebtech(op=lambda x: x2.feval(x))
        result = cb_funcs.multiply(x1_cheb, x2_cheb)
        return trigtech(op=lambda x: result.feval(x))

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
            v = out[0].values
        else:
            c = np.copy(x1.coeffs, order='F')
            v = np.copy(x1.values, order='F')

        # Get the shapes
        fn, fm = x1.shape
        gn, gm = x2.shape

        # prolong
        ncoeffs, nvals = prolong(x1.coeffs, fn + gn - 1, x1.isReal)

        # check dimensions
        if fm != gm:
            if fm == 1:
                # Allow [inf x 1] .* [Inf x m]
                nvals = np.tile(nvals, (1, gm))
                ncoeffs = np.tile(ncoeffs, (1, gm))
            elif gm == 1:
                # Allow [inf x m] .* [Inf x 1]
                x2.values = np.tile(x2.values, (1, fm))
                x2.coeffs = np.tile(x2.coeffs, (1, fm))
            else:
                assert False, 'Inner matrix dimensions must agree!'

        # check for two cases where the output is known in advance to be
        # positive namely F == conj(G) or F == G and isreal(F)
        pos = False

        # multiply values
        if x1 == x2:
            v = np.power(nvals, 2)
            if x1.isreal():
                pos = True

        elif np.conj(x1) == x2:
            v = np.conj(nvals) * nvals
            pos = True
        else:
            gcoeffs, gvals = prolong(x2.coeffs, fn + gn - 1, x2.isReal)
            v = nvals * gvals

        # Compute values and coefficients of result
        c = vals2coeffs(v)

        # Update ishappy
        ishappy = x1.ishappy and x2.ishappy

        # Call simplify
        v, c = simplify_coeffs(c, x1.isReal)

        # TODO
        if pos:
            # when the product should be positive make sure it is after call to simplify
            v = np.abs(v)
            c = vals2coeffs(v)
            is_real = np.ones((1, c.shape[1])).astype(bool)
        else:
            is_real = x1.isreal and x2.isreal

        # if values are real -> make the result real
        v[:, is_real] = np.real(v[:, is_real])
    else:
        return NotImplemented

    # Create the new function
    if out is not None:
        out[0].ishappy = ishappy
        out[0].coeffs = c
        out[0].values = v
        return out[0]
    else:
        return trigtech(values=v, coeffs=c, simplify=True,
                        ishappy=ishappy, isreal=is_real)

def divide(x1, x2, **kwargs):
    return true_divide(x1, x2, **kwargs)

def true_divide(x1, x2, **kwargs):
    # if we have one element that is a ndarray or Number we want that to be x2!
    if isinstance(x1, (np.ndarray, Number)):
        return divide(x2, x1, **kwargs)

    out = kwargs.pop('out', None)
    if out is not None:
        c = out[0].coeffs
        v = out[0].values
    else:
        c = np.copy(x1.coeffs)
        v = np.copy(x1.values)

    if isinstance(x2, Number):
        c /= x2
        v /= x2
        if np.abs(x2) < 1e-16:
            assert False, 'x2 = %.4g' % x2
    elif isinstance(x2, np.ndarray):
        if x1.m == 1:
            c /= x2
            v /= x2
        else:
            c /= np.tile(x2, (x1.n, 1))
            v /= np.tile(x2, (x1.n, 1))
    elif isinstance(x2, type(x1)):
        fun = trigtech(op=lambda x: np.divide(x1(x), x2(x)), minSamples=x1.n + x2.n)
        if out is not None:
            out[0].values = fun.values
            out[0].coeffs = fun.coeffs
            return out[0]
        return fun
    else:
        return NotImplemented

    if out is not None:
        # out[0].values = v
        return out[0]
    else:
        return trigtech(coeffs=c, values=v, ishappy=x1.ishappy, isreal=x1.isReal)
