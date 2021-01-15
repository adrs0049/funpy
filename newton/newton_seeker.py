#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import time
import numpy as np
import scipy
import scipy.linalg as LA
import scipy.sparse.linalg as LAS
import scipy.sparse as sps
from math import sqrt
import warnings
from copy import deepcopy

from funpy.fun import Fun
from funpy.states.deflation_state import DeflationState
from funpy.newton.newton import NewtonBase
from funpy.newton.deflated_residual import DeflatedResidual


class NewtonSeeker(NewtonBase):
    def __init__(self, system, *args, **kwargs):
        super().__init__(system, DeflatedResidual, *args, **kwargs)

        # a list of deflation operators
        self.known_solutions = []

    def to_state(self, coeffs, ishappy=True):
        """ Constructs a new state from coefficients """
        n = coeffs.size // self.n_eqn
        soln = Fun(coeffs=coeffs.reshape((n, self.n_eqn), order='F'),
                   domain=self.system.domain, simplify=False, ishappy=ishappy,
                   type=self.function_type)
        return DeflationState(u=np.real(soln))

    def append(self, soln):
        self.known_solutions.append(soln.u)

    def solve(self, istate, verbose=False, *args, **kwargs):
        """ This function implements the core of the basic newton method """
        success = False
        miter = kwargs.pop('miter', 150)
        learn_solution = kwargs.pop('learn_solution', True)

        # Set shape -> important to how to interpret coefficient vectors!
        self.n_eqn = istate.shape[1]

        # set shape
        self.shape = istate.shape
        self.function_type = 'trig' if istate.u.istrig else 'cheb'

        # Reset the stored outer_v values
        self.cleanup()

        # copy the function TODO proper ctor for deflation from continuation!
        if not isinstance(istate, DeflationState):
            state = DeflationState(u=deepcopy(istate.u), ns=istate.ns)
            try: # FIXME
                state[istate.cpar] = istate.a
            except AttributeError:
                pass
        else:
            state = istate

        # print('Seeker initial state:', state)
        if self.debug: self.iterates.append(deepcopy(state))

        ##### USE A DAMPED NEWTON METHOD: NLEQ_ERR ####
        # Create a mapping that maps u -> D_u F
        LinOp = lambda u: self.linop(u, self.system, ks=self.known_solutions, dtype=self.dtype)
        un, success, iteration = self.nleq_err(LinOp, state, miter=miter,
                                               *args, **kwargs)

        # deflate the operator if we successfully found a solution
        if success and learn_solution:
            if verbose:
                print('NewtonSeeker converged successfully in %d iterations. Inflating residual with:' % iteration, state)
            self.known_solutions.append(un.u)

        return un, success, iteration
