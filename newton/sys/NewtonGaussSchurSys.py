#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
from newton.deflation import deflation, deflation_linearization
from linalg.schur_solve import LinearSolverNGSchur

from newton.lops.NewtonGaussOperator import NewtonGaussOp
from newton.lops.FrechetDeflated import FrechetDeflated
from newton.lops.FrechetDeflatedPrecond import FrechetDeflatedPrecond
from support.tools import findNextPowerOf2, round_up


class NewtonGaussSchurSystem:
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

        # Setup
        self.LinOp = self.matrix(u)
        self.Precond = None

        self.Pr = self.colloc.P

        # Get the number of equations
        self.to_fun = kwargs.get('to_fun', lambda x: x)
        self.to_state = kwargs.get('to_state', lambda x: x)

        # In this case we can use the parameter as the continuation parameter!
        self.solver = LinearSolverNGSchur(self.LinOp, self.Fa,
                                          old_tangent,
                                          self.Pr,
                                          to_function=self.to_fun,
                                          to_state=self.to_state,
                                          eps=self.eps, P=self.Precond)

    @property
    def internal_embedding_parameter(self):
        return -1

    def solve(self, residual, adapt=False):
        return self.solver.solve(residual), True

    def tangent(self):
        return self.solver.tangent_vector()

    def bif_monitor(self, u, old):
        return self.solver.bif_pt(u, self.rvec)

    def bif_dir(self):
        return self.solver.bif_dir()

    def rhs(self, u):
        """ Recomputes the residual at a new state u """
        return self.colloc.rhs(u).squeeze()

    def matrix(self, u):
        M, P, S = self.colloc.matrix()
        return FrechetDeflated(M)

    def precond(self):
        return None
