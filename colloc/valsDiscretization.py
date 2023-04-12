#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
from scipy.sparse import diags

from funpy.cheb import chebpts

from .OpDiscretization import OpDiscretization

HANDLED_FUNCTIONS = {}

class valsDiscretization(OpDiscretization):
    """ Abstract class for collocation discretization of operators. """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.values = np.asarray(kwargs.pop('values', np.zeros(1)))

        # make sure that we have a 2D shape
        if len(self.values.shape) == 1:
            self.values = np.expand_dims(self.values, axis=1)

        self.type = kwargs.get('type', 2)
        assert self.type == 1 or self.type == 2, 'Type must be 1 or 2!'

        # generate a bunch of useful points
        self.x, self.w, self.v, _ = chebpts(self.values.shape[0], interval=self.domain,
                                            type=self.type)
        self.dimension = np.atleast_1d(self.values.shape[0])

    def __call__(self, y):
        """ Evaluates the value collocation at the points y """
        # TODO: somehow detect when we need to map and when we don't!
        # E = self.eval(self.mapping.bwd(y))
        E = self.eval(y)
        return np.dot(E, self.values)

    @property
    def shape(self):
        return self.values.shape

    @property
    def m(self):
        if len(self.shape) == 1:
            return 1
        return self.shape[1]

    def __len__(self):
        return self.values.shape[0]

    def __iter__(self):
        self.ipos = 0
        return self

    def __next__(self):
        if self.ipos >= self.m:
            raise StopIteration
        self.ipos += 1
        return self[self.ipos-1]

    def points(self, pgen):
        dim = np.sum(self.dimension)
        return pgen(dim)

    def mult(self, f):
        """ The multiplications matrix -> for value collocations this is simply the diagonal """
        return diags(f.values)

    def rhs(self):
        """ Discretize the right hand side of a linear system """
        xOut = self.equationPoints()

        blocks = np.empty((1, 1), dtype=object)
        for i in range(1):
            blocks[i, i] = self.eval(blocks[i, i], xOut)

        # create matrix
        b = np.bmat(blocks)

        # prepend the values of the constraints and continuity conditions
        if self.source.constraints:
            b = np.vstack((self.constraint.values, b))

        return b

    def instantiate(self):
        # Move this somewhere else
        M = np.empty((1, 1), dtype=object)
        S = np.empty((1, 1), dtype=object)

        # Currently only has support for one ODE - nothing fancy yet
        for i in range(self.numIntervals):
            for j in range(1):
                M[i, j], S[i, j] = self.quasi2diffmat()

        return M, S

    def getConstraints(self, n):
        nc = len(self.constraints)
        blocks = np.empty(nc, dtype=object)

        for i, constraint in enumerate(self.constraints):
            constraint.domain = self.domain

            # create derivative of the block
            # dfunc = grad(constraint)
            assert False, 'FIXME!'

            # Force float64 here -> otherwise we encounter quite massive rounding errors!
            blocks[i] = np.expand_dims(dfunc(np.zeros((n, self.m), dtype=np.float64)).copy().flatten(order='F'), axis=0)

        return blocks

    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)


""" Decorator to register onp functions """
def implements(np_function):
    def decorator(func):
        HANDLED_FUNCTIONS[np_function] = func
        return func
    return decorator


@implements(np.diff)
def diff(f, k=1, dim=0, *args, **kwargs):
    return f.diff(k=k, axis=dim)
