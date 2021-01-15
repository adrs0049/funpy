#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
from copy import deepcopy
from numbers import Number

from fun import h1norm, norm, norm2, sturm_norm, sturm_norm_alt, sturm_norm
from states.base_state import BaseState
from states.parameter import Parameter


class ContinuationState(BaseState):
    def __init__(self, a=None, u=None, *args, **kwargs):
        # Signature
        signature = [BaseState.SpaceType.REAL, BaseState.SpaceType.CHEB]

        # Call ctor
        super().__init__(signature=signature, *args, **kwargs)

        # Inner product weights
        self.weights = kwargs.pop('weights', np.ones(len(self.signature)))

        # Setup functions
        if a is not None and u is not None:
            self.funcs[0] = deepcopy(u)
            self.reals[0] = deepcopy(a)
            if not isinstance(a, Parameter):
                print('Warning a is not a Parameter!')
                assert False, ''

        # Sync namespace
        self.sync_ns()

    def __array__(self):
        return np.hstack((self.funcs[0].coeffs.flatten(order='F'),
                          self.reals[0].value)).squeeze()

    def __len__(self):
        return self.u.n

    @property
    def u(self):
        return self.funcs[0]

    @property
    def a(self):
        return self.par.value

    @property
    def par(self):
        return self.reals[0]

    @property
    def cpar(self):
        return self.par.name

    def plot_values(self, cidx=0, norm_type='Sturm', p=2, k=1):
        if norm_type == 'Sturm':
            # Computes f'(a) || f ||
            return np.real(self.a), sturm_norm(self.u[cidx], p=p)
        if norm_type == 'Sturm0':
            # Computes f'(a) || f ||
            u0 = self.u - np.sum(np.sum(self.u)) / self.u.shape[1]
            return np.real(self.a), sturm_norm(u0[cidx], p=p)
        elif norm_type == 'SturmNeumann':
            return np.real(self.a), sturm_norm_alt(self.u[cidx], p=p)
        elif norm_type == 'SturmNeumann0':
            # remove the combined mass from the solution state
            u0 = self.u[cidx] - (1. / np.diff(self.u.domain)).item() * np.sum(np.sum(self.u[cidx]))
            return np.real(self.a), sturm_norm_alt(u0, p=p)
        elif norm_type == 'Wkp':
            return np.real(self.a), h1norm(self.u[cidx], p=p, k=k)
        else:
            return np.real(self.a), norm(self.u[cidx], p=p)

    """ Internal interface """
    def __repr__(self):
        return '%s[%s = %.4g, n = %d; |u| = %.4g, ∫u = %.4g]' % \
                (type(self).__name__, self.cpar, np.real(self.a), self.u.shape[0],
                 norm(self.u), self.mass()[0])

    def __str__(self):
        return self.__repr__()
