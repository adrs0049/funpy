#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
import scipy.sparse as sps

from linalg.qr_solve import QR


class FoldSolver:
    """
        / A 0 c1 \ / x \   / f \
        | B A c2 | | y | = | g |
        \ 0 φ 0  / \ z /   \ h /

    """
    def __init__(self, solver, proj, ddx, c1, c2):
        self.solver = solver

        self.c1 = c1
        self.c2 = c2

        self.ddx = ddx
        self.ddx_adj = None
        self.proj = proj

    @property
    def is_singular(self):
        return False

    def solve(self, x):
        """
            Solve this linear system where x denotes the right hand side.

            / A 0  c1 \ / x \   / f \
            | B A  c2 | | y | = | g |
            \ 0 φ* 0  / \ z /   \ h /

            - Note that here φ should be a functional in L2.

            We solve the above system in steps:

                                            ψ^T f
            1) ψ^T (f - c1 z) = 0  =>  z = -------
                                            ψ^T c1

            2) Then solve for the particular solution:

                A x_p = f - z c1

            The general solution is: x = xp + α φ

            3) We determine

                    ψ^T ( g - Bx - z c2 )
                α = ---------------------
                          ψ^T Bφ

            4) Solve for the particular solution:  A y_p = g - Bx - z c2. The general solution is:
                y = yp + β φ, where

                        h - φ^T yp
                   β = -------------
                          φ^T φ

        """
        # Disassemble the right hand size.
        n = x.size
        N = (n - 1) // 2
        f = x[:N]           # Projected onto range?
        g = x[N:2 * N]

        # This is not quite right yet!
        φ, ψ = self.solver.null()   # TODO: need these as functionals! Maybe not?

        # Step 1: Solve first row
        # TODO: Here it really depends which version of f and c1 I use!
        z = np.dot(ψ, f) / np.dot(ψ, self.c1)   # c1 needs to be projected!
        b = self.proj(f - z * self.c1)          # Don't need a projection here
        χ = self.solver.particular(b)

        # Step 2: Use the second line information
        #   The v is the nullspace we are tracking.
        Bχ = self.ddx(χ)
        Bφ = self.ddx(φ)

        # Finally compute the first solution component
        # TODO: same here if ψ the full dual then rhs needs BC
        r = g - z * self.c2 - Bχ           # Here we get a choice, when to project
        α = np.dot(ψ, r) / np.dot(ψ, Bφ)   # Here everything needs to be projected!
        x = χ + α * φ

        # Step 2: Solve the second row
        b = self.proj(r - α * Bφ)
        η = self.solver.particular(b)
        β = (x[n - 1] - np.dot(φ, η)) / np.dot(φ, φ)
        y = η + β * φ

        return np.hstack((x, y, z))

    def solve_adj(self, x):
        """
            Solve this linear system where x denotes the right hand side.

            / A*   B*  0 \ / x \   / f \
            | 0    A*  φ | | y | = | g |
            \ c1* c2*  0 / \ z /   \ h /

            1) Note that here c1 and c2 should be functionals in L2.

            Note any input for solvers to A* we must pass it a function in the
            original basis for the solution vector-space. The solution will then
            "live" in the correct range with respective basis.
        """
        # Disassemble the right hand size.
        n = x.size
        N = (n - 1) // 2
        f = x[:N]           # Projected onto range?
        g = x[N:2 * N]

        # This is not quite right yet!
        φ, ψ = self.solver.null()   # TODO: need these as functionals! Maybe not?

        # Step 1: Solve first row
        # TODO: Here it really depends which version of f and c1 I use!
        z = np.dot(φ, g) / np.dot(φ, φ)    # c1 needs to be projected!
        b = self.proj(g - z * φ)           # Don't need a projection here
        η = self.solver.particular_adj(b)

        # Step 2: Use the second line information
        #   The v is the nullspace we are tracking.
        Bη = self.ddx_adj(η)
        Bψ = self.ddx_adj(ψ)

        # Finally compute the first solution component
        # TODO: same here if ψ the full dual then rhs needs BC
        r = f - Bη                        # Here we get a choice, when to project
        α = np.dot(φ, r) / np.dot(φ, Bψ)  # Here everything needs to be projected!
        y = η + α * ψ

        # Step 2: Solve the second row
        b = self.proj(r - α * Bψ)
        χ = self.solver.particular_adj(b)
        β = (x[n - 1] - np.dot(self.c1, x) - np.dot(self.c2, η)) / np.dot(self.c2, ψ)
        x = χ + β * ψ

        return np.hstack((x, y, z))
