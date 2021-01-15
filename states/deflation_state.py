#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
from copy import deepcopy
from numbers import Number

from funpy.fun import h1norm, norm, norm2, sturm_norm, sturm_norm_alt, sturm_norm
from funpy.states.base_state import BaseState
from funpy.states.parameter import Parameter


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

    def __len__(self):
        return self.u.n

    @property
    def u(self):
        return self.funcs[0]

    def plot_values(self, cidx=0, p=2, k=1):
        return sturm_norm(self.u[cidx], p=p)

    """ Internal interface """
    def __repr__(self):
        return '%s[n = %d; |u| = %.4g, ∫u = %.4g]' % \
                (type(self).__name__, self.u.shape[0], norm(self.u), self.mass()[0])

    def __str__(self):
        return self.__repr__()
