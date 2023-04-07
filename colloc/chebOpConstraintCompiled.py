#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np

from cheb.detail import polyval
from fun import Fun
from colloc.chebcolloc.chebcolloc2 import chebcolloc2


class ChebOpConstraintCompiled:
    def __init__(self, source, *args, **kwargs):
        self.dimension = np.atleast_1d(kwargs.get('dimension', 1))
        self.domain = np.asarray(kwargs.get('domain', [-1, 1]))
        self.source = source
        self.n = kwargs.pop('n', 2**15)
        self.m = kwargs.pop('m', 2)

        self.constraints = self.source.constraints if self.source is not None else []

    @property
    def shape(self):
        # TODO: temp for the moment!
        return (self.n, self.m)

    """ Number of constraints in object """
    def __len__(self):
        return self.functional.shape[0]

    def compile(self):
        # number of constraints
        nc = len(self.constraints)

        # number of equations
        n, m = self.shape

        # Generate a value collocation to first represent the functionals
        fun = Fun(op=m * [lambda x: np.zeros_like(x)], type='cheb')
        fun.prolong(n)
        valColloc = chebcolloc2(self.source, values=fun.values, domain=self.domain)

        # get the constraint matrices
        vConstraints = valColloc.getConstraints(n)

        # Need to apply the Fourier transform trick per equation element!
        blocks = np.empty(nc, dtype=object)
        for i, constraint in enumerate(vConstraints):
            blocks[i] = np.zeros_like(constraint)

            for k in range(m):
                Ml = np.rot90(constraint[:, k*n:(k+1)*n] / self.n).astype(np.double)
                blocks[i][:, k*n:(k+1)*n] = self.n * np.rot90(polyval(Ml), -1)

        return blocks
