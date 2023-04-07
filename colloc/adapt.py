#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
from support.tools import findNextPowerOf2
from copy import deepcopy
import numpy as np


class SolverAdaptive:
    """ Class encapsulating a solver that solves collocation systems on
    nested increasing Chebyshev grids.

    TODO: remove the duplication of essentially this code with inside the continuation code.
    """
    def __init__(self, nw, *args, **kwargs):
        """
        Arguments: nw -> an instance of a nonlinear solver.
        """
        self.ntol = kwargs.pop('tol', 1e-8)
        self.n_min = min(7, kwargs.pop('n_min', 2))
        self.n_max = kwargs.pop('n_max', 14)
        self.nw = nw

    def n_disc(self, lower_bd=17, n_trans=9, constant=False, *args, **kwargs):
        """ The discretization sizes the solver uses.

        Arguments:
            lower_bd  int   discretization sizes must be above this number.
            constant  bool  We are looking for constant solutions only -> disc size is 1.

            For heterogeneous solutions the minimum is 17.
        """
        if constant:
            return np.array([1])

        upper_bd = kwargs.pop('upper_bd', 1 + 2**self.n_max)
        return np.unique(np.maximum(lower_bd,
                                    np.minimum(upper_bd, 1 + 2**np.arange(self.n_min, 1 + self.n_max))))

    def solve(self, state, eps=1e-8, up_steps=12, max_fail_steps=3,
              verbose=False, learn_solution=False, *args, **kwargs):
        """ The solver method

            Arguments:
                state -> Must be of BaseState type.
        """
        lower_bd = state.shape[0]
        upper_bd = 2**int(np.log2(findNextPowerOf2(lower_bd)) + max(up_steps - 1, 0))
        n_its = self.n_disc(lower_bd=lower_bd, upper_bd=upper_bd, *args, **kwargs)
        success = np.zeros_like(n_its).astype(bool)

        for k, n in enumerate(n_its):
            if state is not None: state.prolong(n)
            self.nw.system.setDisc(n)

            # Solve the system! -> Since we repeat we cannot learn solutions without causing
            # subsequent convergence failure!
            nst, success[k], it = self.nw.solve(state, learn_solution=learn_solution,
                                                tol=self.ntol, *args, **kwargs)

            # Compute the difference between the two most recent approximations. If this difference
            # is small we are happy with the solution and return it!.
            d_norm = (state - nst).norm() if state is not None else np.inf

            if verbose:
                print(f'\tSolve with n = {n}; k = {k}; iters = {it}; neval = {self.nw.neval}; |d| = {d_norm:.6e}; success = {success[:k+1]}')

            # We only update the state variables if Newton's method was successful!
            if success[k]:
                if d_norm < eps and k > 0:
                    break

                state = deepcopy(nst)
                continue

            # If we have already done 3 iterations that have failed; let's bail!
            elif k >= max_fail_steps - 1 and np.all(~success):
                break

        # The n required for the solution
        return state, success[k], it
