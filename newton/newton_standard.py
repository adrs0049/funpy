#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
import scipy

from states.deflation_state import DeflationState
from newton.newton import NewtonBase
from newton.deflated_residual import DeflatedResidual


class Newton(NewtonBase):
    def __init__(self, system, *args, **kwargs):
        super().__init__(system, DeflatedResidual, *args, **kwargs)

        # a list of deflation operators
        self.known_solutions = []

    def to_state(self, coeffs, ishappy=True):
        """ Constructs a new state from coefficients """
        soln = self.to_fun(coeffs)
        return DeflationState(u=np.real(soln))

    def solve(self, u0, verbose=False, method='nleq', *args, **kwargs):
        """ This function implements the core of the basic newton method """
        miter = kwargs.pop('miter', 75)
        assert isinstance(u0, DeflationState), ''

        # number of equations
        self.n_eqn = u0.shape[1]
        self.shape = (u0.n, u0.m)
        self.function_type = 'trig' if u0.istrig else 'cheb'

        # Reset the stored outer_v values
        self.cleanup()

        ##### USE A DAMPED NEWTON METHOD: NLEQ_ERR ####
        # Create a mapping that maps u -> D_u F
        LinOp = lambda u: self.linop(u, self.system, ks=self.known_solutions, dtype=self.dtype)

        if method == 'qnerr':
            return self.qnerr(LinOp, u0, miter=miter, *args, **kwargs)
        elif method == 'nleq':
            return self.nleq_err(LinOp, u0, miter=miter, *args, **kwargs)
