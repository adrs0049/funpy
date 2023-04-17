#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np

from .detail import polyfit, polyval
from .pts import chebpts_type2_compute


class FunctionContainer:
    def __init__(self, fs, dtype=np.float64, *args, **kwargs):
        self.fs = fs
        self.dtype = dtype

    def __len__(self):
        return len(self.fs)

    def __call__(self, x):
        r = np.empty((x.size, len(self.fs)), order='F', dtype=self.dtype)
        for i, f in enumerate(self.fs):
            r[:, i] = f(x)
        return r


""" Base class for resampling and refining operations  """
class RefineBase:
    def __init__(self, op, *args, **kwargs):
        self.minSamples = max(9, kwargs.pop('minSamples', 9))
        self.op = op
        self.strategy = kwargs.pop('strategy', 'nested')
        self.values = np.zeros((0,0), order='F')

    def __call__(self, target):
        return self._call(target)

    """ Guess the next domain size to use """
    def get_n(self, f):
        if f.size == 0:
            n = 2**np.ceil(np.log2(self.minSamples - 1)) + 1
        else:
            pow = np.log2(f.shape[0] - 1)
            if pow == np.floor(pow) and pow > 5:
                n = np.round(2**(np.floor(pow) + 0.5)) + 1
                n = n - np.remainder(n, 2) + 1
            else:
                n = 2**(np.floor(pow) + 1) + 1

        if n > f.maxLength:
            if self.values is None:
                n = f.maxLength
                giveUp = False
            else:
                giveUp = True
        else:
            giveUp = False

        # TODO: make sure that n is not too large
        return int(n), giveUp

""" Class to refine polynomials in values """
class Refine(RefineBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.strategy == 'nested':
            self._call = self.__nested
        elif self.strategy == 'resample':
            self._call = self.__resample
        else:
            assert False, ''

    """ Resample the function on a new domain """
    def __resample(self, target):
        n, giveUp = self.get_n(target)

        if giveUp:
            return self.values, giveUp

        x = chebpts_type2_compute(n)
        self.values = np.real(self.op(x)).astype(np.float64)
        if self.values.ndim == 1:
            self.values = np.expand_dims(self.values, axis=1)
        return self.values, giveUp

    """ Resample the function on a nested domain """
    def __nested(self, target):
        if self.values.size == 0:
            return self.__resample(target)

        n = 2 * self.values.shape[0] - 1

        if n > target.maxLength:
            giveUp = True
            return self.values, giveUp
        else:
            giveUp = False

        x = chebpts_type2_compute(n)
        # take every 2nd entry
        x = x[1:-1:2]

        # shift the stored values
        new_values = np.zeros((n, self.values.shape[1]), order='F')
        new_values[0:n:2, :] = self.values

        # compute the new values XXX fix this!
        nv = np.real(self.op(x))
        if nv.ndim == 1:
            nv = np.expand_dims(nv, axis=1)
        new_values[1:-1:2, :] = nv

        self.values = new_values
        return self.values, giveUp

""" Class to resample existing chebyshev polynomials """
class RefineCompose1(RefineBase):
    def __init__(self, f, op, *args, **kwargs):
        super().__init__(op, *args, **kwargs)
        self.f = f
        self._call = self.__nested

    def __resample(self, target):
        n, giveUp = self.get_n(target)

        if giveUp:
            return self.values, giveUp

        # update f-values
        self.f.prolong(n)
        v1 = polyval(self.f.coeffs)

        # compute the new values
        self.values = self.op(v1)
        return self.values, giveUp

    def __nested(self, target):
        if self.values.size == 0:
            self.values, giveUp = self.__resample(target)
        else:
            n = 2 * target.shape[0] - 1

            if n > target.maxLength:
                giveUp = True
                return self.values, giveUp
            else:
                giveUp = False

            # check that n is too large
            self.f.prolong(n)
            fvalues = polyval(self.f.coeffs)
            v1 = fvalues[1:-1:2, :]

            # shift the values
            new_values = np.zeros((n, self.values.shape[1]), order='F')
            new_values[0:n:2, :] = self.values

            # compute the new values
            new_values[1:-1:2, :] = self.op(v1)
            self.values = new_values

        return self.values, giveUp

""" Class to re-sample existing Chebyshev polynomials """
class RefineCompose2(RefineBase):
    def __init__(self, f, op, g, *args, **kwargs):
        super().__init__(op, *args, **kwargs)
        self.f = f
        self.g = g
        self._call = self.__nested

    def __resample(self, target):
        n, giveUp = self.get_n(target)
        if giveUp:
            return self.values, giveUp

        # update f-values
        self.f.prolong(n)
        v1 = polyval(self.f.coeffs)
        self.g.prolong(n)
        v2 = polyval(self.g.coeffs)

        # compute the new values
        self.values = self.op(v1, v2)
        return self.values, giveUp

    def __nested(self, target):
        if self.values.size == 0:
            self.values, giveUp = self.__resample(target)
        else:
            n = 2 * target.shape[0] - 1

            if n > target.maxLength:
                giveUp = True
                return self.values, giveUp
            else:
                giveUp = False

            # check that n is too large
            self.f.prolong(n)
            fvalues = polyval(self.f.coeffs)
            v1 = fvalues[1:-1:2, :]
            self.g.prolong(n)
            gvalues = polyval(self.g.coeffs)
            v2 = gvalues[1:-1:2, :]

            # shift the values
            new_values = np.zeros((n, self.values.shape[1]), order='F')
            new_values[0:n:2, :] = self.values

            # compute the new values
            new_values[1:-1:2, :] = self.op(v1, v2)
            self.values = new_values

        return self.values, giveUp
