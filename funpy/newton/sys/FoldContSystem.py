#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np

from linalg.qr_solve import QR
from linalg.bem_solve import BEM
from linalg.bemw_solve import BEMW

from fun import Fun, prolong, normalize
from functional import Functional

from states.mp_state import MoorePenroseState
from states.tp_cont_state import TwoParContState

from support.tools import findNextPowerOf2, round_up


class NewtonGaussFoldSystem:
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
    def __init__(self, mp_state, operator, phi, psi, *args, **kwargs):
        # State is a MoorePenrose state u -> ContPar; v -> the tangent
        # The derivatives wrt parameter
        self.Fa = None
        self.Fb = None

        # state
        state = mp_state.u
        tangent = mp_state.v

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

        # Update the tangent
        self.update_tangent()

    @property
    def size(self):
        return np.product(self.shape) + 1

    def tangent(self):
        return (self.t + self.dt).normalize()

    def to_vspace(self, coeffs, *args, **kwargs):
        u = TwoParContState.from_coeffs(coeffs, self.bpar, self.cpar, self.n_eqn, domain=self.domain)
        return MoorePenroseState(u=u, v=self.dt)

    def update_tangent(self):
        """
            Since (for the moment at least) we must use full Newton
            iterations at each step. We will update the tangent according
            to the standard Moore-Penrose approximation. In more detail:

            1. B = [ F_x(X_k) V_k ]
            2. R = [ AV_k 0 ]
            3. W = V_k - B^-1 R      <-> correction is a zero solve.
            4. V_(k+1) = W / || W || <-> taken care of in MoorePenroseState

            Here A is given by:

                / f_x f_α f_β \ / Δx \
                |             | | Δα |
                \ h_x h_α h_β / \ Δβ /

        """
        # Computes the tangent correction at this step.
        dt = self.fsolver.solve_zero(np.hstack((self.Rx, self.Ra)))
        self.dt = -TwoParContState.from_coeffs(dt, self.bpar, self.cpar, self.n_eqn, domain=self.domain)

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
        Next create the low row of the Frechet derivative of F(x, α, β):

             / f_x f_α f_β \ / Δx \
        dF = |             | | Δα |
             \ h_x h_α h_β / \ Δβ /

        The full Moore-Penrose continuation matrix will then be:

             / f_x f_α f_β \ / Δx \
        dH = | h_x h_α h_β | | Δα |
             \ V_x V_α V_β / \ Δβ /

        and then determine the min-norm solution of this matrix to advance in β.
        V is the tangent vector i.e. spans the null-space of matrix dF. The resulting
        matrix dH is a double bordered matrix.

        The top row is easily calculated. For the bottom row we employ the
        following ``trick''.

            - h_x = -w^T f_xx v
            - h_α = -w^T f_xα v
            - h_β = -w^T f_xβ v

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
        h_sign = -1.0 * self.h_sign_corr

        # Store these we will look them up
        self.v = normalize(v)
        self.w = normalize(w)

        # Compute the parameter derivatives and project them.
        Fa = self.colloc.project_vector(source.pDer(u, pname=u.bname))
        Fb = self.colloc.project_vector(source.pDer(u, pname=u.cpar))

        # This is the bilinear action we need
        hx = source.ddx(u.u, v, w)
        hx = h_sign * np.asarray(Functional(hx, n=self.n)).squeeze()

        # Project to range space since w is a functional on this space
        ha = h_sign * np.dot(w, source.dxdp(u, v, pname=u.bname))
        hb = h_sign * np.dot(w, source.dxdp(u, v, pname=u.cpar))

        # Compute a tangent vector.
        Vx = np.asarray(Functional(t.u, n=self.n)).squeeze()
        Va = t.b
        Vb = t.a  # This is the cpar

        # Compute tangent rhs for tangent update
        uc = t.u.flatten()
        self.Rx = M.dot(uc) + Fa * Va + Fb * Vb
        self.Ra = np.dot(hx, uc) + ha * Va + hb * Vb

        # Create the full solver matrix
        self.fsolver = BEMW(self.csolver,
                            [hx, np.hstack((Vx, Va))],
                            [Fa, np.hstack((Fb, hb))],
                            [ha, Vb])

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
            The nonlinear system is:

                / F(x, α, β) \
                \ g(x, α, β) /

            where g = h.
        """
        # TODO: Can we do better than this?
        colloc = self.op.discretize(u, par=True)
        M, _, P = colloc.matrix()

        # This is the only way? TODO: replace QR with a sparse LU solver!
        solver = QR(M)
        v, h = self.solve_brd(solver, self.phi, self.psi)

        res = self.colloc.project_vector(self.source.rhs.values.flatten(order='F'), cts=True)
        return np.hstack((res, h))
