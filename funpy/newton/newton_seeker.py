#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
import time
from math import sqrt
import warnings
from copy import deepcopy

from fun import Fun
from newton.nleq_err import nleq_err
from newton.giant import giant_gbit
from states.deflation_state import DeflationState
from newton.newton import NewtonBase
from newton.sys.NewtonSystem import NewtonSystem


class NewtonSeeker(NewtonBase):
    def __init__(self, system, *args, **kwargs):
        super().__init__(system, NewtonSystem, *args, **kwargs)

        # TODO? FIXME FIXME
        self.n_eqn = system.neqn
        self.domain = system.domain

        # The transformation functions
        self.to_fun   = lambda c: Fun.from_coeffs(c, self.n_eqn, domain=self.domain)
        self.to_state = lambda c: DeflationState.from_coeffs(c, self.n_eqn, domain=self.domain)

        # a list of deflation operators
        self.known_solutions = []

    def append(self, solns):
        if isinstance(solns, list):
            self.known_solutions.extend(solns)
        else:
            self.known_solutions.append(solns.u)

    def solve(self, istate, verbose=False, method='giant', *args, **kwargs):
        """ This function implements the core of the basic newton method """
        miter = kwargs.pop('miter', 150)
        learn_solution = kwargs.pop('learn_solution', True)

        # Set shape -> important to how to interpret coefficient vectors!
        self.n_eqn = istate.shape[1]
        self.shape = istate.shape
        self.function_type = 'trig' if istate.u.istrig else 'cheb'

        # Create a mapping that maps u -> D_u F
        LinOp = lambda u, *args, **kwargs: self.linop(
            u, self.system, ks=self.known_solutions,
            dtype=self.dtype, *args, **kwargs)

        method = method.lower()
        if method == 'giant':
            un, success, iteration, self.neval, sys = giant_gbit(
                LinOp, istate, miter=miter, inner=np.dot,
                *args, **kwargs)

        elif method == 'nleq' or method == 'nleq-err':
            un, success, iteration, self.neval, sys = nleq_err(
                LinOp, istate, miter=miter, inner=np.dot,
                *args, **kwargs)

        else:
            raise RuntimeError(f'Unknown solver method {method}!')

        # deflate the operator if we successfully found a solution
        if success and learn_solution:
            if verbose:
                print('NewtonSeeker converged successfully in %d iterations. Inflating residual with:' % iteration, istate)
            self.known_solutions.append(un.u)

        # Store for lookup
        self.linsys = sys

        return un, success, iteration
