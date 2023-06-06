#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
from copy import deepcopy

from ..fun import Fun, norm, sturm_norm
from .base_state import BaseState


class DeflationState(BaseState):
    def __init__(self, u=None, *args, **kwargs):
        signature = [BaseState.SpaceType.CHEB]
        super().__init__(signature=signature, *args, **kwargs)

        # Inner product weight
        self.weights = kwargs.pop('weights', np.ones(len(self.signature)))

        # Setup functions - TODO implement np.equal for state!
        if u is not None:
            self.funcs[0] = deepcopy(u)

        # Sync namespace
        self.sync_ns()

    @classmethod
    def from_state(cls, other, **kwargs):
        pars = {n: v for n, v in other.items()}
        return cls(u=other.u, **{**pars, **kwargs})

    @classmethod
    def from_coeffs(cls, coeffs, n_funs, *args, **kwargs):
        assert coeffs.size % n_funs == 0, f'New coefficients with size {coeffs.size} does not fit expected {n_funs}!'
        soln = Fun.from_coeffs(coeffs, n_funs, *args, **kwargs)
        return cls(u=np.real(soln))

    @classmethod
    def from_fun(cls, fun, *args, **kwargs):
        return cls(u=np.real(fun))

    @property
    def u(self):
        return self.funcs[0]

    def __len__(self):
        return self.u.n

    def plot_values(self, cidx=0, p=2, k=1):
        return sturm_norm(self.u[cidx], p=p)

    """ Internal interface """
    def __repr__(self):
        return '%s[n = %d; |u| = %.4g, ∫u = %.4g]' % \
                (type(self).__name__, self.u.shape[0], norm(self.u), self.mass()[0])

    def __str__(self):
        return self.__repr__()
