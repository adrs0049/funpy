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
from sparse.csr import eliminate_zeros_csr

from funpy.fun import minandmax
from funpy.fun import Fun, h1norm, norm, norm2
from funpy.cheb.chebpts import quadwts
from funpy.cheb.diff import computeDerCoeffs
from funpy.states.State import ContinuationState
from funpy.states.deflation_state import DeflationState
from funpy.newton.pseudo_arclength import PseudoArcContinuationCorrector
from funpy.newton.newton_gauss import NewtonGaussContinuationCorrector
from funpy.newton.newton import NewtonBase
from funpy.newton.deflated_residual import DeflatedResidual
from funpy.linalg.qr_solve import QRCholesky
from nlep.nullspace import right_null_vector
from funpy.support.tools import orientation_y, logdet, functional, Determinant

class Newton(NewtonBase):
    def __init__(self, system, *args, **kwargs):
        super().__init__(system, DeflatedResidual, *args, **kwargs)

        # a list of deflation operators
        self.known_solutions = []

    def to_state_defl(self, coeffs, ishappy=True):
        """ Constructs a new state from coefficients """
        n = coeffs.size // self.n_eqn
        soln = Fun(coeffs=coeffs.reshape((n, self.n_eqn), order='F'),
                   domain=self.system.domain, ishappy=ishappy, type=self.function_type)
        return DeflationState(u=np.real(soln), n=n)

    def to_state_cont(self, coeffs, ishappy=True):
        """ Constructs a new state from coefficients """
        print('to state cont')
        m = coeffs.size // self.n_eqn
        if m * self.n_eqn == coeffs.size:
            return self.to_fun(coeffs, ishappy=ishappy)

        soln = self.to_fun(coeffs[:-1], ishappy=ishappy)
        state = ContinuationState(a=np.real(coeffs[-1]), u=np.real(soln),
                                  cpar=self.cpar, n=max(1, soln.shape[0]), theta=self.theta)
        return state

    def to_fun(self, coeffs, ishappy=True):
        """ Constructs a new state from coefficients """
        m = coeffs.size // self.n_eqn
        soln = Fun(coeffs=coeffs.reshape((m, self.n_eqn), order='F'), simplify=False,
                   domain=self.system.domain, ishappy=ishappy, type=self.function_type)
        return soln

    def solve(self, u0, verbose=False, method='nleq', *args, **kwargs):
        """ This function implements the core of the basic newton method """
        miter = kwargs.pop('miter', 75)

        if type(u0) == DeflationState:
            self.to_state = self.to_state_defl
        elif type(u0) == ContinuationState:
            self.to_state = self.to_state_cont
        else:
            assert False, ''

        # number of equations
        self.n_eqn = u0.shape[1]
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
