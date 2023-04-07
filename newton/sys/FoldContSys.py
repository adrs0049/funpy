#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np

from linalg.qr_solve import QR
from linalg.bem_solve import BEM
from linalg.bemw_solve import BEMW
from linalg.fold_solve import FoldSolver

from fun import Fun, prolong, normalize
from functional import Functional

from states.tp_state import TwoParameterState

from support.tools import findNextPowerOf2, round_up


class NewtonGaussFoldExtendedSystem:
    """
        Represents the Newton system to solve the equations

                     / f(x, α, β) \
        F(x, α, β) = |            | = 0
                     \ h(x, α, β) /

        where
            - f(x, α, β) is the nonlinear function
            - h(x, α, β) defined below:

            / f_x  b \ / v \    / 0 \
            |        | |   |  = |   |
            \ c^T  0 / \ h /    \ 1 /

            - Obvious choices for b is the left null-vector and
            for c the right null-vector.

            - b, c will be old right and left null-vectors respectively
            - v is proportional to the right null-vector of f_x.
            - w is proportional to the left null-vector of f_x
            - h is expected to be zero when f_x is singular.
            - This linear system will be solved using the bordering technique.

            - TODO: make sure we don't redo the LU / QR decompositions here!

        The Jacobian of F(x, α, β) is given by

             / f_x f_α f_β \ / Δx \
        dF = |             | | Δα |
             \ h_x h_α h_β / \ Δβ /

        The top row is easily calculated. For the bottom row we employ the
        following ``trick''.

            - h_x = -w^T f_xx v
            - h_α = -w^T f_xα v
            - h_β = -w^T f_xβ v

    """
    def __init__(self, state, operator, tangent, phi, psi, *args, **kwargs):
        # State is a MoorePenrose state u -> ContPar; v -> the tangent
        # The derivatives wrt parameter
        self.Fa = None
        self.Fb = None

        # state
        # state = mp_state.u
        # tangent = mp_state.v

        # set discretization size to next higher power of 2.
        n = state.shape[0] - 1
        self.n = round_up(n, 64) + 1 if n >= 64 else findNextPowerOf2(n) + 1
        self.n = kwargs.pop('n', 5)
        # print('N = ', self.n)
        operator.setDisc(self.n)

        self.domain = state.domain
        self.n_eqn = state.shape[1]
        self.to_fun = lambda c: Fun.from_coeffs(c, self.n_eqn, domain=self.domain)

        self.cpar = state.cpar
        self.bpar = state.bname

        # Create shape
        self.shape = (self.n, operator.neqn)
        self.h_sign_corr = kwargs.pop('h_sign', 1.0)

        # Create discretization
        self.op = operator
        self.colloc = operator.discretize(state, par=True)
        self.source = self.colloc.source

        # The right and left null-vectors
        self.phi = prolong(phi, self.n)
        self.psi = prolong(psi, self.n)

        # print('φ = ', np.asarray(self.phi))
        # print('ψ = ', np.asarray(self.psi))

        # Special values: see comments below.
        self.d   = kwargs.pop('d', 0.0)
        self.t   = tangent

        self.M = None
        self.N = None

        # The core solver for the discretization
        self.csolver = None

        # The full solver for the fold continuation
        self.fsolver = None

        # Setup
        self.matrix(state, self.phi, self.psi, tangent)

    @property
    def size(self):
        return 2 * np.product(self.shape) + 1

    def tangent(self):
        """
            Compute the tangent of the map.
        """
        nt = self.fsolver.solve_null()
        return self.to_vspace(nt).normalize()

    def to_vspace(self, coeffs, *args, **kwargs):
        return TwoParameterState.from_coeffs(coeffs, self.bpar, self.cpar, self.n_eqn, domain=self.domain)

    def solve_brd(self, solver, phi, psi, *args, **kwargs):
        """
        The bordered matrix:

            / A  c \ / x \   / 0 \
            |      | |   | = |   |
            \ b  d / \ z /   \ 1 /

        For the matrix above to be nonsingular we need that:
            1. C spans a subspace transversal to R[A] i.e. N[A^T]
            2. B spans a subspace transversal to R[A^T] i.e. N[A]

        To construct the matrix with the smallest condition number we set:
            1. d = 0
            2. B, C, to be orthonormal sets of the above basis.

        Remarks:
            1. When z = 0, then A is singular.
            2. When A is singular then x is a multiple of the null-vector.

        Dual problem:

            / A^T  c \ / x \   / 0 \
            |        | |   | = |   |
            \ b    d / \ z /   \ 1 /

        For the matrix above to be nonsingular we need that:
            1. C spans a subspace transversal to R[A^T] i.e. N[A]
            2. B spans a subspace transversal to R[A] i.e. N[A^T]

            N[A]   == (u, v)
            N[A^T] == (BC0, BC1, BC2, BC3, u, v)

        To construct the matrix with the smallest condition number we set:
            1. d = 0
            2. B, C, to be orthonormal sets of the above basis.
        """
        b = np.asarray(Functional(phi, n=self.n)).squeeze()
        c = self.colloc.project_vector(psi.flatten())
        solver = BEM(solver, b, c, self.d)

        # Solve the system as above
        v = solver.solve_null()
        h = v[-1]

        # Map solutions to function objects.
        v = normalize(self.to_fun(v[:-1]))
        return v, h

    def matrix(self, u, phi, psi, t, *args, **kwargs):
        """

            u: TwoParameterState
                -> u -> the current function state
                -> φ -> the current nullspace

        """
        source = self.colloc.source
        M, N, P = self.colloc.linop()

        # Create core solver
        self.csolver = QR(M)
        self.asolver = QR(N)

        # Compute the core problem - not ideal but without the explicitly
        # constructed adjoint operator the transpose solver returns a dirac
        # delta derivative distribution most likely originating from the boundary rows.
        v, h1 = self.solve_brd(self.csolver, phi, psi)
        w, h2 = self.solve_brd(self.asolver, psi, phi)

        # Store these we will look them up
        self.v = normalize(v)
        self.w = normalize(w)

        # Compute the parameter derivatives.
        Fa  = source.pDer(u, pname=u.bname)
        Fb  = source.pDer(u, pname=u.cpar)
        Fxa = source.dxdp(u, v, pname=u.bname)
        Fxb = source.dxdp(u, v, pname=u.cpar)

        # TODO: Check that Fa in Range
        psiFa = np.dot(w, Fa)
        psiFb = np.dot(w, Fb)

        # TODO: flip if one of those is zero!

        # Create a functional for the bilinear form
        ddx  = lambda w: source.ddx(u.u, v, w)
        proj = lambda x: self.colloc.project_vector(x, cts=True)

        # Assemble the core solver
        csolver = FoldSolver(self.csolver, proj, ddx, Fa, Fxa)

        print('t = ', t)
        Vx = t
        Vb = t.b

        # Assemble the full solver -> need the tangent here
        # TODO: check where we need the various projections!
        self.fsolver = BEM(csolver, Vx, np.hstack((Fb, Fxb)), Vb)

    def solve(self, residual, *args, **kwargs):
        """
        Solve the system

           / f_x f_α f_β \ / Δx \   / F(x) \
           | h_x h_α h_β | | Δα | = | g(x) |
           \ V_x V_α V_β / \ Δβ /   \  0   /

        Denote this system as B Δy = Q
        """
        Δy = self.fsolver.min_norm(residual)
        return self.to_vspace(Δy), True

    def rhs(self, u):
        """
        The right hand side of the equation is

            /  F(X)  \
            | F_x[v] |
            \   1    /
        """
        # TODO: do we really project here?
        r1 = self.source.rhs.values.flatten(order='F')

        # TODO: do we really project here?
        r2 = self.source.dx(u.u, u.v)

        return np.hstack((r1, r2, 1.0))
