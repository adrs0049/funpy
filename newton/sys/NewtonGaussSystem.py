#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
from newton.deflation import deflation, deflation_linearization
from linalg.qr_chol import QRCholesky
from linalg.lsmr_solve import LSMR

from newton.lops.NewtonGaussOperator import NewtonGaussOp
from support.tools import findNextPowerOf2, round_up


class NewtonGaussLinearSystem:
    """
        Approximates a linear operator in a function space.
    """

    def __init__(self, u, operator, old_tangent, rvec, switch_point=0.30,
                 par=False, *args, **kwargs):
        # Tolerances
        self.eps = kwargs.pop('eps', 1e-10)
        self.rvec = rvec

        # The derivative wrt parameter
        self.Fa = None

        # set discretization size to next higher power of 2.
        n = u.shape[0] - 1
        N = round_up(n, 64) + 1 if n >= 64 else findNextPowerOf2(n) + 1
        operator.setDisc(N)

        # Create shape
        self.shape = (N, operator.neqn)

        # Create discretization
        self.colloc = operator.discretize(u, par=par)

        # FIXME
        self.source = self.colloc.source

        # Setup
        self.LinOp = self.matrix(u)
        self.Precond = None

        # Get the number of equations
        self.to_fun = kwargs.get('to_fun', lambda x: x)
        self.to_state = kwargs.get('to_state', lambda x: x)

        # The tangent is almost vertical -> use QR-Cholesky
        self.solver = QRCholesky(self.LinOp.to_matrix().todense(),
                                 rank=self.LinOp.shape[0])

        #self.solver2 = LSMR(self.LinOp)

    @property
    def internal_embedding_parameter(self):
        return self.solver.internal_embedding_parameter

    def solve(self, residual, adapt=False, *args, **kwargs):
        """ Solve the linear system! """
        x = self.solver.solve(residual)
        func = self.to_fun(x[:-1])
        x = self.to_state(func, x[-1])
        return x, True

    def tangent(self):
        t = self.solver.tangent_vector()
        func = self.to_fun(t[:-1])
        t = self.to_state(func, t[-1])
        return t

    def bif_monitor(self, u, old):
        return self.solver.det()

    def bif_dir(self):
        return None

    def rhs(self, u):
        """ Recomputes the residual at a new state u """
        self.source.update_partial(u)
        return self.colloc.project_vector(self.source.rhs.values.flatten(order='F'),
                                          basis=True,
                                          cts=True)

    def matrix(self, u):
        # M, MT, P    = self.colloc.linop()
        M, _, P     = self.colloc.matrix()
        Fa          = self.source.pDer(u)
        self.Fa_bc  = self.colloc.project_vector(Fa)
        #self.Fa_adj = np.concatenate((np.zeros(self.colloc.numConstraints), self.colloc.convertToFunctional(Fa)))
        #return NewtonGaussOp(M, MT, self.Fa_bc, self.Fa_adj)

        return NewtonGaussOp(M, None, self.Fa_bc, None)

    def precond(self):
        return None
