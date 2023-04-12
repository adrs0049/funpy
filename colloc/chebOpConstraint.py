#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np

from colloc.chebcolloc.chebcolloc2 import chebcolloc2


class ChebOpConstraint:
    """
        Implements a constraint for an cheb operator.
    """
    def __init__(self, *args, **kwargs):
        """
            op: The constraint: a callable.
            ns: A local namespace: No longer used.
            domain: [x0, x1]. The domain on which parent operator is defined ->
                required for constraint collocation.

            values: Equal value of the constraint. TODO: drop this!
        """
        self.functional = kwargs.get('op', None)  # applied to the variable to get values
        self.ns = kwargs.pop('ns', None)
        self.values     = np.atleast_1d(kwargs.get('values', 0.0))  # constraint on the result of the functional
        # TODO: see whether we can get rid of this silly domain requirement
        self.domain = kwargs.pop('domain', [-1, 1])
        self.compiled = None

    def append(self, func, value=0):
        self.functional = np.vstack((self.functional, func))
        self.values = np.vstack((self.values, value))

    def disc(self, u):
        return chebcolloc2(values=u, domain=self.domain)

    """ Number of constraints in object """
    def __len__(self):
        return self.functional.shape[0]

    def __neg__(self):
        self.values = -self.values
        return self

    def __call__(self, u):
        # def __call__(self, *args): -> args : [u, v]
        # generate a value collocation representation
        colloc = chebcolloc2(values=u, domain=self.domain)
        return self.functional(*colloc).squeeze()

    def residual(self, u):
        """ Compute the residual of the functional """
        return self.functional(*u) - self.values
