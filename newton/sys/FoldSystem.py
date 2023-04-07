#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np

from linalg.qr_solve import QR
from linalg.bem_solve import BEM
from linalg.bemw_solve import BEMW

from fun import Fun, prolong, normalize
from functional import Functional
from states.cont_state import ContinuationState

from support.tools import findNextPowerOf2, round_up


class FoldSystem:
    """
        Represents the Newton system to solve the equations

                   / f(x, α) \
        F(x, α) = |          | = 0
                   \ h(x, α) /

        where
            - f(x, α) is the nonlinear function
            - h(x, α) defined below:

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

        The Jacobian of F(x, α) is given by

             / f_x f_α \ / Δx \
        dF = |         | |    |
             \ h_x h_α / \ Δα /

        The top row is easily calculated. For the bottom row we employ the
        following ``trick''.

            - h_x = -w^T f_xx v
            - h_α = -w^T f_xα v

    """
    def __init__(self, state, operator, phi, psi, *args, **kwargs):

        # The derivative wrt parameter
        self.Fa = None

        # set discretization size to next higher power of 2.
        n = state.shape[0] - 1
        self.n = round_up(n, 64) + 1 if n >= 64 else findNextPowerOf2(n) + 1
        self.n = kwargs.pop('n', 9)
        operator.setDisc(self.n)

        self.domain = state.domain
        self.n_eqn = state.shape[1]
        self.to_fun = lambda c: Fun.from_coeffs(c, self.n_eqn, domain=self.domain)

        self.cpar = state.cpar

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

        # Special values: see comments below.
        self.d   = kwargs.pop('d', 0.0)

        # The core solver for the discretization
        self.csolver = None

        # The full solver for the fold continuation
        self.fsolver = None

        # Setup
        self.matrix(state, self.phi, self.psi)

        # Update the tangent
        # self.update_tangent()

    @property
    def size(self):
        return np.product(self.shape) + 1

    def to_vspace(self, coeffs, *args, **kwargs):
        return ContinuationState.from_coeffs(coeffs, self.cpar, self.n_eqn, domain=self.domain)

    def update_tangent(self):
        """
            Since (for the moment at least) we must use full Newton
            iterations at each step. We will update the tangent according
            to the standard Moore-Penrose approximation. In more detail:

            1. B = [ F_x(X_k) V_k ]
            2. R = [ AV_k 0 ]
            3. W = V_k - B^-1 R      <-> correction is a zero solve.
            4. V_(k+1) = W / || W ||

            Here A is given by:

                / f_x f_α \ / Δx \
                |         | |    |
                \ h_x h_α / \ Δα /

        """
        self.phi = self.v
        self.psi = self.w

        # Computes the tangent correction at this step.
        # dt = self.fsolver.solve_zero(np.hstack((self.Rx, self.Ra)))
        # self.dt = ContinuationState.from_coeffs(dt, self.bpar, self.cpar, self.n_eqn, domain=self.domain)

        # self.t -= self.dt
        # self.t.normalize()

        # nt = self.fsolver.solve_null()
        # self.t = self.to_vspace(nt).normalize()

        # Update the tangent vector
        # Vx = np.asarray(Functional(self.t.u, n=self.n)).squeeze()
        # Va = self.t.b
        # Vb = self.t.a  # This is the cpar!

        # Update the row in the BRD solver
        # self.fsolver.update_row(1, np.hstack((Vx, Va)), Vb)

        # print('Post update = ')
        # print(self.fsolver.mat)

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
        v = self.to_fun(v[:-1])
        return v, h

    def matrix(self, u, phi, psi, *args, **kwargs):
        """
        Next create the low row of the Frechet derivative of F(x, α):

             / f_x f_α \ / Δx \
        dF = |         | |    |
             \ h_x h_α / \ Δα /

        The top row is easily calculated. For the bottom row we employ the
        following ``trick''.

            - h_x = -w^T f_xx v
            - h_α = -w^T f_xα v

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

        # Compute the parameter derivatives and project them.
        Fa = self.colloc.project_vector(source.pDer(u, pname=u.cpar))

        # This is the bilinear action we need
        hx = source.ddx(u.u, v, w)
        hx = h_sign * np.asarray(Functional(hx, n=self.n)).squeeze()

        # Project to range space since w is a functional on this space
        print('u = ', u)
        print('v = ', v)
        print('w = ', w)
        r = source.dxdp(u, v, pname=u.cpar)
        print('r = ', r)
        ha = h_sign * np.dot(w, source.dxdp(u, v, pname=u.cpar))

        # Store these we will look them up
        self.v = normalize(v)
        self.w = normalize(w)

        # Create the full solver matrix
        self.fsolver = BEM(self.csolver, hx, Fa, ha)

    def solve(self, residual, *args, **kwargs):
        """
        Solve the system

             / f_x f_α \ / Δx \    / F(x) \
             |         | |    |  = |      |
             \ h_x h_α / \ Δα /    \ g(x) /

        Denote this system as B Δy = Q
        """
        Δy = self.fsolver.solve(residual)
        return self.to_vspace(Δy), True

    def rhs(self, u):
        """
            The nonlinear system is:

                / F(x, α) \
                \ g(x, α) /

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
