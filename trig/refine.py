#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
from trig.trigpts import trigpts

def expand(array, axis=1):
    if len(array.shape) == 1:
        return np.expand_dims(array, axis=axis)
    return array

""" Base class for resampling and refining operations  """
class RefineBase(object):
    def __init__(self, op, *args, **kwargs):
        self.minSamples = kwargs.pop('minSamples', 9)
        self.op = op
        self.strategy = kwargs.pop('strategy', 'nested')
        self.values = np.zeros((0,0), dtype='complex')

    def __call__(self, target):
        return self._call(target)

    """ Guess the next domain size to use """
    def get_n(self, f):
        if f.size == 0:
            n = 2**np.ceil(np.log2(self.minSamples - 1))
        else:
            pow = np.log2(f.shape[0])
            if pow == np.floor(pow) and pow > 5:
                n = 3*2**(pow-1)
            else:
                n = 2**(np.floor(pow) + 1)

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


""" Class to refine trig polynomials in values """
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

        x, _ = trigpts(n)
        x = np.hstack((x, 1))
        self.values = expand(self.op(x))

        # compute the average values of f at -/+ 1 and then remove the +1 value
        self.values[0, :] = 0.5 * (self.values[0, :] + self.values[-1, :])
        self.values = self.values[:-1, :]
        return self.values, giveUp

    """ Resample the function on a nested domain """
    def __nested(self, target):
        if self.values.size == 0:
            return self.__resample(target)

        n = 2 * self.values.shape[0]

        if n > target.maxLength:
            giveUp = True
            return self.values, giveUp
        else:
            giveUp = False

        x, _ = trigpts(n)
        # take every 2nd entry
        x = x[1::2]

        # shift the stored values
        new_values = np.zeros((n, self.values.shape[1]), dtype=self.values.dtype)
        new_values[:n:2, :] = self.values

        # compute the new values
        new_values[1::2, :] = expand(self.op(x))
        self.values = new_values
        return self.values, giveUp
