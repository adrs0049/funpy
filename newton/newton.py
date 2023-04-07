#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
from math import sqrt

import states.base_state as sp

from newton.newton_base import NewtonBase, NewtonErrors
from newton.newton_standard import newton_standard
from newton.giant import giant_gbit
from newton.qnerr import qnerr
from newton.nleq_err import nleq_err

from newton.sys.NewtonSystem import NewtonSystem


class Newton(NewtonBase):
    def __init__(self, system, *args, **kwargs):
        sys = kwargs.pop('sys', NewtonSystem)
        super().__init__(system, sys, *args, **kwargs)

        # a list of deflation operators
        self.known_solutions = []

    def callback(self, system, itr, dx, normdx, thetak):
        self.normdxs.append(normdx)
        return True

    def solve(self, u0, verbose=False, op_kwargs={}, *args, **kwargs):
        """ This function implements the core of the basic newton method """
        # Set function type
        self.function_type = 'trig' if u0.istrig else 'cheb'

        # Create a mapping that maps u -> D_u F
        method = kwargs.pop('method', 'giant').lower()
        linearSolver = {'qnerr': 'lu', 'nleq': 'lu', 'nleq-err': 'lu', 'giant': None}

        def LinOp(u, *args, **kwargs):
            return self.linop(u, self.system, ks=self.known_solutions,
                              dtype=self.dtype,
                              exact=linearSolver.get(method, method),
                              **op_kwargs)

        # Newton solver selection
        if method == 'qnerr' or method == 'qn-err':
            x, success, k, self.normdx, self.neval, sys = qnerr(
                LinOp, u0, inner=np.dot, dtype=self.dtype, callback=self.callback,
                *args, **kwargs)

            self.status = NewtonErrors.Success if success else NewtonErrors.NonlinSolFail

        elif method == 'nleqerr' or method == 'nleq' or method == 'nleq-err':
            x, success, k, self.normdx, self.neval, sys = nleq_err(
                LinOp, u0, dtype=self.dtype, callback=self.callback,
                inner=np.dot, *args, **kwargs)

            self.status = NewtonErrors.Success if success else NewtonErrors.NonlinSolFail

        elif method == 'giant' or method == 'giant-gbit' or method == 'giantgbit':
            x, success, k, self.neval, sys = giant_gbit(
                LinOp, u0, inner=np.dot, dtype=self.dtype, callback=self.callback,
                *args, **kwargs)

            self.status = NewtonErrors.Success if success else NewtonErrors.NonlinSolFail

        else:  # Standard newton method!
            x, success, k, self.normdx, self.neval, sys = newton_standard(
                LinOp, u0, inner=np.dot, dtype=self.dtype, callback=self.callback,
                *args, **kwargs)

            self.status = NewtonErrors.Success if success else NewtonErrors.NonlinSolFail

        # Store for lookup
        self.linsys = sys

        return x, success, k
