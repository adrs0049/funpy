#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np

import scipy.linalg as LA

from newton.deflation import deflation, deflation_linearization
from linalg.qr_solve import QR
from linalg.bem_solve import BEM

from fun import normalize
from functional import Functional
from newton.lops.NewtonGaussOperator import NewtonGaussOp
from support.tools import findNextPowerOf2, round_up

from states.cont_state import ContinuationState


class MoorePenroseSystem:

    def __init__(self, state, operator, tangent, rvec, *args, **kwargs):
        """
        Arguments:

            - point:     Point on the continuation curve.
            - operator:  The linear operator

        """
        # Tolerances
        self.eps = kwargs.pop('eps', 1e-10)
        self.op  = operator
        self.sp  = kwargs.pop('switch_point', 0.3)

        # Store some more values
        self.cpar   = state.cpar
        self.domain = operator.domain
        self.n_eqn  = operator.neqn

        # To be set
        self.colloc = None
        self.source = None

        # Store tangent
        self.t    = tangent
        self.rvec = rvec
        self.bdir = None

        # Setup
        self.Fx, self.Fa, self.Vx, self.Va = self.discretize(operator, state, tangent)

        # Create shape
        self.shape = (self.n, self.op.neqn)

        # Create the core and full solver
        self.csolver = QR(self.Fx)
        self.fsolver = BEM(self.csolver, self.Vx, self.Fa, self.Va)

        # Update the tangent
        self.update_tangent()

        # Keep track of singular status
        self.is_singular(tangent)

    @property
    def size(self):
        return np.product(self.shape)

    @property
    def internal_embedding_parameter(self):
        """
            If Δα is zero, that means that Fx is singular. In this case,
            we must be using some other internal embedding parameter.
            For the moment we will simply select the element with the
            largest absolute value. This should also persist between steps.
        """
        return self.chi

    def to_vspace(self, coeffs, *args, **kwargs):
        return ContinuationState.from_coeffs(coeffs, self.cpar, self.n_eqn, domain=self.domain)

    def det(self):
        """
            By Cramer's rule the solution of the system

            / Fx  Fα \ / φ \   / 0 \
            |        | |   | = |   |
            \ Vx  Vα / \ s /   \ 1 /

            we have that:

                        det(A)
                    s = -----
                        det(M)

            Since the total matrix M is nonsingular we have that s(A) has
            the same sign as det(A), further φ will be a scalar multiple
            of the null-vector of Fx when Fx is singular. Further:

                [ φ s ] is the nullspace of [ Fx Fα ]

        """
        # Check whether the linearization is singular
        return self.fsolver.solve_null()[-1]

    def is_singular(self, state):
        """
            Check whether the linearization is singular.
        """
        self.chi = np.argmax(np.abs(state.u.coeffs.flatten(order='F'))) if abs(self.Va) < self.sp else -1

    def tangent(self):
        return self.t

    def left_nullvector(self):
        null = self.fsolver.solve_null_adj()
        r, nt = self.colloc.toFunctionIn(null[:-1])
        nt = ContinuationState.from_fun(nt, self.cpar, null[-1], domain=self.domain)
        return nt.normalize()

    def update_tangent(self):
        """
            Update the tangent vector.
        """
        nt = self.fsolver.solve_null()
        self.t = self.to_vspace(nt).normalize()

        # Update the tangent vector
        self.Vx = np.asarray(Functional(self.t.u, n=self.n)).squeeze()
        self.Va = self.t.a

        # Update the row in the BRD solver
        self.fsolver.update_row(self.Vx, self.Va)

    def solve(self, residual, *args, **kwargs):
        """
        Solve the system

            / Fx  Fα \ / Δx \   / F(x) \
            |        | |    | = |      |
            \ Vx  Vα / \ Δα /   \  0   /

        Denote this system as B Δy = Q
        """
        # Compute the application of the Moore-Penrose inverse of the top row.
        Δy = self.fsolver.min_norm(residual)
        return self.to_vspace(Δy), True

    def rhs(self, u):
        """
            Returns the vector: F(u)
        """
        self.source.update_partial(u)
        return self.colloc.project_vector(self.source.rhs.values.flatten(order='F'), cts=True)

    def discretize(self, op, state, tangent):
        # set discretization size to next higher power of 2.
        n = state.shape[0] - 1
        self.n = n
        # self.n = round_up(n, 64) + 1 if n >= 64 else findNextPowerOf2(n) + 1
        self.n = max(self.n, 2**3 + 1) if n > 1 else 1
        op.setDisc(self.n)

        # Create discretization
        self.colloc = op.discretize(state, par=True)
        self.source = self.colloc.source

        M, _, P = self.colloc.matrix()
        Fa      = self.source.pDer(state).prolong(self.n)
        Fa_bc   = self.colloc.project_vector(Fa)
        Vx      = np.asarray(Functional(tangent.u, n=self.n)).squeeze()
        Va      = tangent.a

        # TEMP
        self.Faa = Fa

        return M, Fa_bc, Vx, Va

    def update_func(self):
        """
        test: complete me

        """
        return

        # 1. Update Bu
        c = self.rvec.C_u.flatten()
        c = self.colloc.project_vector(self.rvec.C_u.prolong(self.n).flatten())
        Bn = self.solver.solve_adj(np.hstack((c, self.rvec.C_l)))
        r, B = self.colloc.toFunctionIn(Bn[:-1])
        B = normalize(B).prolong(self.n)

        # 2. Update Cu
        Cn = self.solver.solve(np.hstack((self.colloc.project_vector(self.rvec.B_u.prolong(self.n)), self.rvec.B_l)))

        # Update rvec
        self.rvec.C_u = normalize(self.to_fun(Cn[:-1])).prolong(self.n)
        self.rvec.C_l = Cn[-1]
        self.rvec.B_u = B
        self.rvec.B_l = Bn[-1]

    def bif_monitor(self, *args, **kwargs):
        """
            Employ the bordering technique once more, but now for the
            Moore-Penrose continuation matrix itself, to determine
            when that matrix is singular i.e. when we have a potential
            branch point.

            / Fx  Fα Bu \ / Yx \   / 0 \
            | Vx  Vα Bα | | Yα | = | 0 |
            \ Cx  Cα  0 / \  τ /   \ 1 /

            where:
                - J is the block matrix in the upper left
                - B not in R[J]   i.e. N[J^T]
                - C not in R[J^T] i.e. N[J]

            TODO: Should we just call a version of BEMW? and then solver for null?

        """
        return 1.0

        # TODO replace all this mess
        if self.rvec.counter == 0:
            self.update_func()
        self.rvec.counter += 1

        # Get Wu from the solver
        Wu = self.to_fun(self.solver.beta)

        # Step 1: Compute beta
        beta = self.t.a - np.dot(self.t.u, Wu)

        # Compute Pu
        Pu = self.solver.qr.solve(self.colloc.project_vector(self.rvec.B_u))
        Pu = self.to_fun(Pu)

        # Step 2: Compute Gl
        Gl = (self.rvec.B_l - np.dot(self.t.u, Pu)) / beta

        # Step 3: Compute Gu
        Gu = Pu - Gl * Wu

        # Step 4: Compute tau
        tau = - 1.0 / (self.rvec.C_l * Gl + np.dot(self.rvec.C_u, Gu))

        # TODO: verify me.
        # det = self.solver.det()
        # print('\tTau = ', tau, ' det(J) = ', det)

        # Step 5: Compute potential bifurcation direction
        self.bdir = - tau * self.to_state(Gu, Gl)
        return tau

    def bif_dir(self):
        return self.bdir.normalize()
