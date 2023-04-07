#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
import scipy.sparse as sps

from linalg.qr_solve import QR


class BEM:
    """
        Solve the matrix

            / A c \  / x \  =  / f \
            |     |  |   |  =  |   |
            \ b d /  \ z /  =  \ h /

        using the mixed block elimination method BEM.

        For the matrix above to be nonsingular we need that:
            1. C spans a subspace transversal to R[A] i.e. N[A^T]
            2. B spans a subspace transversal to R[A^T] i.e. N[A]

        To construct the matrix with the smallest condition number we set:
            1. d = 0
            2. B, C, to be orthonormal sets of the above basis.

    """
    def __init__(self, solver, b, c, d):
        """
            Solver must be a class that provides ``some'' methods.
        """
        # self.A  = A
        # self.sparse = sps.issparse(A)

        self.solver = solver
        self.b  = b
        self.c  = c
        self.d  = d

        # If not singular do the pre-processing steps for BEM.
        if not self.is_singular: self.prepare()

    @property
    def is_singular(self):
        return self.solver.is_singular

    @property
    def shape(self):
        return (self.solver.shape[0] + 1, self.solver.shape[1] + 1, )

    @property
    def mat(self):
        n, m = self.solver.shape
        if self.sparse:
            op = sps.bmat([[self.A, np.atleast_2d(self.c).T], [np.atleast_2d(self.b), self.d]])
        else:
            op = np.empty(self.shape, dtype=float)
            op[:n, :m] = self.A
            op[m, :m]  = self.b
            op[:n, m]  = self.c
            op[n, m]   = self.d
        return op

    def det(self):
        return self.solver.det() * (self.d - np.dot(self.b, self.beta))

    def prepare(self):
        # Do the pre-processing steps
        # 1. Pre-processing for BED
        self.gamma = self.solver.solve_adj(self.b)   # b is in N[A] --> not in R[A^T]
        self.delta = self.d - np.dot(self.gamma, self.c)

        # 2. Pre-processing for BEC
        self.beta  = self.solver.solve(self.c)       # c is in N[A^T] --> not in R[A]
        self.eta   = self.d - np.dot(self.b, self.beta)

    def update_row(self, b, d):
        self.b = b
        self.d = d

        # Update the pre-processing steps
        if not self.is_singular:
            self.gamma = self.solver.solve_adj(self.b)   # b is in N[A] --> not in R[A^T]
            self.delta = self.d - np.dot(self.gamma, self.c)
            self.eta   = self.d - np.dot(self.b, self.beta)

    def update_col(self, c, d):
        self.c = c
        self.d = d

        # Update the pre-processing steps
        if not self.is_singular:
            self.delta = self.d - np.dot(self.gamma, self.c)
            self.beta  = self.solver.solve(self.c)       # c is in N[A^T] --> not in R[A]
            self.eta   = self.d - np.dot(self.b, self.beta)

    def nullspace(self):
        # BED approximation to y
        y1 = 1.0 / self.delta

        # Residual correction by BEC
        w  = -self.solver.solve(y1 * self.c)
        y2 = (y1 * self.d + np.dot(self.b, w)) / self.eta
        return np.hstack((w + self.beta * y2, y1 - y2))

    def nullspace_adj(self):
        # BED approximation to y
        y1 = 1.0 / self.eta

        # Residual correction by BEC
        w  = -self.solver.solve_adj(y1 * self.b)
        y2 = (y1 * self.d + np.dot(self.c, w)) / self.delta
        return np.hstack((w + self.gamma * y2, y1 - y2))

    def solve_singular(self, b):
        """
            Solve the system when the matrix A is singular. We proceed in steps:

                1. Solve for the particular solution of

                    A xp = f - z c

                    where z = (ψ, b) / (ψ, c) guaranteeing that the rhs is in R[A].

                2. x = xp + α φ

                    where α = (b[-1] - d z - (b, xp)) / (b, φ)

                so that the lowest row is satisfied.
        """
        # Get the null-vectors
        phi, psi = self.solver.null()

        # Solve the system via the bordering technique
        z = np.dot(psi, b[:-1]) / np.dot(psi, self.c)

        # Get the particular solution
        xp = self.solver.particular(b[:-1] - z * self.c)
        al = (b[-1] - self.d * z - np.dot(self.b, xp)) / np.dot(self.b, phi)
        return np.hstack((xp + al * phi, z))

    def solve_singular_adj(self, b):
        """ Do the same as above, just for the adjoint of A """
        # Get the null-vectors
        phi, psi = self.solver.null()

        # Solve the system via the bordering technique
        z = np.dot(phi, b[:-1]) / np.dot(phi, self.b)

        # Get the particular solution
        xp = self.solver.particular_adj(b[:-1] - z * self.b)
        al = (b[-1] - self.d * z - np.dot(self.c, xp)) / np.dot(self.c, psi)
        return np.hstack((xp + al * psi, z))

    def solve_nonsing(self, b):
        # BED approximation to y
        y1 = (b[-1] - np.dot(self.gamma, b[:-1])) / self.delta

        # Residual correction by BEC
        w  = self.solver.solve(b[:-1] - y1 * self.c)
        y2 = (b[-1] - y1 * self.d - np.dot(self.b, w)) / self.eta
        return np.hstack((w - self.beta * y2, y1 + y2))

    def solve_nonsing_adj(self, b):
        # Flip beta <-> gamma; eta <-> delta
        # BED approximation to y
        y1 = (b[-1] - np.dot(self.beta, b[:-1])) / self.eta

        # Residual correction by BEC
        w  = self.solver.solve_adj(b[:-1] - y1 * self.b)
        y2 = (b[-1] - y1 * self.d - np.dot(self.c, w)) / self.delta
        return np.hstack((w - self.gamma * y2, y1 + y2))

    def solve(self, b):
        return self.solve_singular(b) if self.is_singular else self.solve_nonsing(b)

    def solve_adj(self, b):
        return self.solve_singular_adj(b) if self.is_singular else self.solve_nonsing_adj(b)

    def solve_null_nonsing(self):
        """ Solves the system with f = 0; g = 1 """
        y1 = 1.0 / self.delta
        y2 = (1.0 + y1 * (np.dot(self.b, self.beta) - self.d)) / self.eta
        return (y1 + y2) * np.hstack((-self.beta, 1))

    def solve_null_nonsing_adj(self):
        """ Solves the adjoint system with f = 0; g = 1 """
        y1 = 1.0 / self.eta
        y2 = (1.0 + y1 * (np.dot(self.c, self.gamma) - self.d)) / self.delta
        return (y1 + y2) * np.hstack((-self.gamma, 1))

    def solve_null_singular(self):
        # Get the null-vectors
        phi, psi = self.solver.null()
        return np.hstack((phi, 0))

    def solve_null_singular_adj(self):
        # Get the null-vectors
        phi, psi = self.solver.null()
        return np.hstack((psi, 0))

    def solve_null(self):
        return self.solve_null_singular() if self.is_singular else self.solve_null_nonsing()

    def solve_null_adj(self):
        return self.solve_null_singular_adj() if self.is_singular else self.solve_null_nonsing_adj()

    def solve_zero_nonsing(self, f):
        """ Solves the system with f not 0; g = 0 """
        # BED approximation to y
        y1 = -np.dot(self.gamma, f) / self.delta

        # Residual correction by BEC
        w  = self.solver.solve(f - y1 * self.c)
        y2 = - (y1 * self.d + np.dot(self.b, w)) / self.eta
        return np.hstack((w - self.beta * y2, y1 + y2))

    def solve_zero_singular(self, f):
        """ Solves the system with f not 0; g = 0 """
        # Get the null-vectors
        phi, psi = self.solver.null()

        # Solve the system via the bordering technique
        z = np.dot(psi, f) / np.dot(psi, self.c)

        # Get the particular solution
        xp = self.solver.particular(f - z * self.c)
        al = - (self.d * z + np.dot(self.b, xp)) / np.dot(self.b, phi)
        return np.hstack((xp + al * phi, z))

    def solve_zero(self, f):
        return self.solve_zero_singular(f) if self.is_singular else self.solve_zero_nonsing(f)

    def min_norm(self, f):
        """
            Find the minimum norm solution of the underdetermined system

                [ A c ] (α β)^T = f      (1)

            We solve the system:

            / A c \  / x \  =  / f \
            |     |  |   |  =  |   |
            \ b d /  \ y /  =  \ 0 /

            and

            / A c \  / u \  =  / 0 \
            |     |  |   |  =  |   |
            \ b d /  \ v /  =  \ 1 /

        Then the general solution of the system (1) is given by:

            1. α = x + η u
            2. β = y + η v

        The minimum norm solution is in R[A^T] i.e. it must be orthogonal to N[A].
        Using this we derive the following condition:

                     x^T u + y v
            η = - ------------------
                    u^T u + v^T v

        """
        x   = self.solve_zero(f)
        t   = self.solve_null()
        eta = np.dot(x, t) / np.dot(t, t)
        return x - eta * t
