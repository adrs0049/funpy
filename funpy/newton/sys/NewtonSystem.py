#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
import scipy.sparse.linalg as LAS

from funpy.linalg.exact_linear_solver import ExactLinearSolver
from funpy.vectorspaces import DeflationState

from ..deflation import deflation, deflation_linearization
from ..lops.FrechetDeflated import FrechetDeflated
from ..lops.FrechetDeflatedPrecond import FrechetDeflatedPrecond


class NewtonSystem:
    """
        Approximates a linear operator in a function space.
    """

    def __init__(self, u, operator, exact=None, par=False, *args, **kwargs):
        # Store the known solutions here
        self.known_solutions = kwargs.get('ks', [])

        # Create discretization
        self.colloc = operator.discretize(u, par=par)
        self.source = self.colloc.source

        # The possible linear system
        self.linSys = None
        self.n_eqn = operator.neqn
        self.domain = u.domain
        self.dtype = kwargs.get('dtype')
        self.ftype = 'trig' if self.dtype == np.complex128 else np.float64

        # Create shape
        self.shape = (operator.n_disc, self.n_eqn)

        # Place holders for deflation information
        self.u = u
        self.res = None
        self.eta = None
        self.deta = None

        # If exact -> this will create an exact linear solver to solve the system
        if exact is not None:
            self._create_exact_linear_solver(exact)

    @property
    def size(self):
        return np.product(self.shape)

    def to_vspace(self, coeffs, *args, **kwargs):
        return DeflationState.from_coeffs(coeffs, self.n_eqn, domain=self.domain, type=self.ftype)

    def _create_exact_linear_solver(self, linear_solver, precond=False):
        linOp, precd, proj = self.matrix(precond=precond)

        # Create the exact linear solver
        self.linSys = ExactLinearSolver(linOp, proj=lambda x: proj.dot(x),
                                        method=linear_solver)

    def solve(self, b):
        assert self.linSys is not None, 'Solving via sparse method don\'t call this!'

        # Solve the linear system!
        x, success = self.linSys.solve(b)
        return self.to_vspace(x), success

    def rhs(self, u=None):
        """ Recomputes the residual at a new state u """
        self.source.update_partial(u)
        res = self.colloc.project_vector(self.source.rhs.values.flatten(order='F'), cts=True)

        if not self.known_solutions:
            return res

        return deflation(u.u, known_solutions=self.known_solutions) * res

    def matrix(self, precond=False):
        M, _, P = self.colloc.matrix()

        if self.known_solutions:
            self.res = self.colloc.project_vector(self.source.rhs.values.flatten(order='F'), cts=True)
            self.eta, self.deta = deflation_linearization(self.u.u, known_solutions=self.known_solutions)

        # Create the matrix and its precond
        proj   = self.colloc.proj
        matrix = FrechetDeflated(M, eta=self.eta, b=self.res, deta=self.deta)

        # Use a rough estimate of the linear operator as the preconditioner
        if precond:
            M2, _, P2 = self.colloc.matrix(eps=1e-2)
            Prec = LAS.splu(M2.tocsc())

            precnd = FrechetDeflatedPrecond(
                Prec, self.res, ks=len(self.known_solutions),
                functional=self.deta, eta=self.eta)
        else:
            precnd = None

        return matrix, precnd, proj
