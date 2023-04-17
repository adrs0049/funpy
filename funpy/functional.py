#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np

from .cheb import polyval
from .cheb import quadwts

HANDLED_FUNCTIONS = {}


class Functional:
    """
        Turns a function into a functional.
    """
    def __init__(self, function, *args, **kwargs):
        weighted = kwargs.pop('weighted', False)
        rescaleFactor = 0.5 * np.diff(function.domain)

        n = kwargs.pop('n', None)
        if n is not None:
            function = function.prolong(n)

        if weighted:
            self.coeffs = np.asarray(function)
            self.coeffs[0] *= np.pi
            self.coeffs[1:] *= 0.5 * np.pi

        else:
            functional = np.empty(function.m, dtype=object)
            w = quadwts(function.n)
            for j, col in enumerate(function):
                functional[j] = np.rot90(polyval(np.rot90(w * col.values.T)), -1)

            self.coeffs = np.hstack(functional) * rescaleFactor

    def __array__(self):
        return self.coeffs

    """ Implement array ufunc support """
    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        # TODO: why does this not work!
        #if not all(isinstance(t, self.__class__) or issubclass(t, Fun) for t in types):
        #    print('class = ', self.__class__)
        #    print('t = ', types)
        #    return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)


def implements(np_function):
    """ Register an __array_function__ implementation """
    def decorator(func):
        HANDLED_FUNCTIONS[np_function] = func
        return func
    return decorator


@implements(np.dot)
def dot(input1, input2):
    return np.dot(np.asarray(input1), np.asarray(input2))
