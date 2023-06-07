#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np

from ..fun import zeros
from .chebcolloc.chebcolloc2 import chebcolloc2


class ChebOpConstraint:
    """
        Implements linear functional which represent constraints
        for ChebOp operators.
    """
    def __init__(self, *args, **kwargs):
        """
            op: The constraint: a callable.
            ns: A local namespace: No longer used.
            domain: [x0, x1]. The domain on which parent operator is defined ->
                required for constraint collocation.

            values: Equal value of the constraint.
        """
        self.op     = kwargs.get('op', None)
        self.domain = kwargs.pop('domain', [-1, 1])
        self.values = np.atleast_1d(kwargs.get('values', 0.0))

        # Assemble dummy function
        zf = zeros(1, domain=self.domain)
        offset = -self.residual(zf)

        def functional(*args):
            return self.op(*args) + offset

        self.functional = functional

    def append(self, func, value=0):
        self.functional = np.vstack((self.functional, func))
        self.values     = np.vstack((self.values, value))

    def __len__(self):
        """ Number of constraints in object """
        return self.functional.shape[0]

    def __neg__(self):
        self.values = -self.values
        return self

    def __call__(self, n):
        colloc = chebcolloc2(values=np.eye(n), domain=self.domain)
        return self.functional(colloc)

    def residual(self, u):
        """ Compute the residual of the functional """
        return self.op(*u) - self.values
