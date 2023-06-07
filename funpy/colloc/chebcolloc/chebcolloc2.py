#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
import scipy.linalg as LA

from ...cheb import chebtech
from ...cheb import chebpts_type1, chebpts_type2
from ...cheb import barymat, diffmat

from ..valsDiscretization import valsDiscretization


HANDLED_FUNCTIONS = {}


class chebcolloc2(valsDiscretization):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x, _, self.v, _ = self.functionPoints()

    def __getitem__(self, key):
        return chebcolloc2(values=self.values[:, key], domain=self.domain)

    @property
    def tech(self):
        return chebtech

    def toValues(self):
        " Converts a chebtech2 to values at 2nd kind points"""
        pass

    def __call__(self, y):
        """ Returns the evaluation functional for the operator """
        E = self.eval(y)
        return E @ self.values

    def eval(self, locations):
        """ Evaluation functional for chebcolloc

            return a functional that evaluates the chebyshev polynomial
            represented by a colloc directization at the given point loc
        """
        locations = np.atleast_1d(locations)
        assert locations >= self.domain[0] and locations <= self.domain[1], \
                'Evaluation point %.4g must be inside the domain %s!' % (locations, self.domain)
        return barymat(locations, self.x, self.v)

    def functionPoints(self):
        return self.points(lambda N: chebpts_type2(N, interval=self.domain))

    def equationPoints(self):
        return self.points(lambda N: chebpts_type1(N, interval=self.domain))

    def diffmat(self, k=1, axis=0, **unused_kwargs):
        domain = self.domain
        n = self.dimension
        if k == 0:
            return np.eye(np.sum(n))

        # assuming that we only have on interval
        blocks = np.empty(self.numIntervals, dtype=object)
        for i in range(self.numIntervals):
            length = domain[i+1] - domain[i]
            # Don't have to scale already done in diffmat
            blocks[i] = diffmat(self.x, k=k) # * (2/length)**k

        return LA.block_diag(*blocks)

    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)


def implements(np_function):
    """ Decorator to register np functions """
    def decorator(func):
        HANDLED_FUNCTIONS[np_function] = func
        return func
    return decorator


@implements(np.diff)
def diff(f, n=1, axis=0, *args, **kwargs):
    D = f.diffmat(k=n, axis=axis, **kwargs)
    return chebcolloc2(values=D @ f.values, domain=f.domain)


@implements(np.sum)
def sum(f, *args, **kwargs):
    pass

