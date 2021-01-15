#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as onp

from funpy.fun import Fun
from funpy.colloc.chebcolloc.chebcolloc2 import chebcolloc2

# Import auto differentiation
import jax.numpy as np

class ChebOpConstraint(object):
    def __init__(self, *args, **kwargs):
        self.functional = kwargs.get('op', None)  # applied to the variable to get values
        self.values     = onp.atleast_1d(kwargs.get('values', 0.0))  # constraint on the result of the functional
        # TODO: see whether we can get rid of this silly domain requirement
        self.domain = kwargs.pop('domain', [-1, 1])
        self.compiled = None

    def append(self, func, value=0):
        self.functional = onp.vstack((self.functional, func))
        self.values = onp.vstack((self.values, value))

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
