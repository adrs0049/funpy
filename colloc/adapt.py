#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
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
        self.tol = kwargs.pop('tol', 1e-8)
        self.ntol = kwargs.pop('ntol', 1e-8)
        self.ltol = kwargs.pop('ltol', 1e-10)
        self.n_min = min(7, kwargs.pop('n_min', 4))
        self.n_max = kwargs.pop('n_max', 11)
        self.nw = nw

    def n_disc(self, lower_bd=17, constant=False, *args, **kwargs):
        """ The discretization sizes the solver uses.

        Arguments:
            lower_bd  int   discretization sizes must be above this number.
            constant  bool  We are looking for constant solutions only -> disc size is 1.

            For heterogeneous solutions the minimum is 17.
        """
        if constant:
            return np.array([1])

        return np.unique(np.maximum(lower_bd, 1 + 2**np.arange(self.n_min, self.n_max)))

    def solve(self, state, verbose=False, *args, **kwargs):
        """ The solver method

            Arguments:
                state -> Must be of BaseState type.
        """
        n_its = self.n_disc(*args, **kwargs)
        success = np.zeros_like(n_its).astype(bool)

        new_state = None
        for k, n in enumerate(n_its):
            if new_state is None:
                init_state = deepcopy(state)
                init_state.prolong(n)
            else:
                init_state = deepcopy(new_state)
                init_state.prolong(n)

            if new_state is not None: new_state.prolong(n)
            self.nw.system.setDisc(n)

            # Solve the system! -> Since we repeat we cannot learn solutions without causing
            # subsequent convergence failure!
            nst, success[k], it = self.nw.solve(init_state, learn_solution=False, scale=False,
                                                tol=self.ntol, inner_tol=self.ltol, *args, **kwargs)

            # Compute the difference between the two most recent approximations. If this difference
            # is small we are happy with the solution and return it!.
            d_norm = (new_state - nst).norm() if new_state is not None else np.inf

            if verbose:
                print('\tSolve with n = %d; k = %d; iters = %d; neval = %d; |d| = %.4g, |r| = %.4g; success = '
                      % (n, k, it, self.nw.neval, d_norm, self.nw.normfk), success[:k+1])

            # We only update the state variables if Newton's method was successful!
            if success[k]:
                if d_norm < 1e-5:
                    break

                new_state = nst
                continue
            else:
                new_state = None

        # The n required for the solution
        return new_state, success[k], it
