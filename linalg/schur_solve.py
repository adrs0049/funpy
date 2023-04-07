#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
import scipy as sp
from math import sqrt
import scipy.sparse as sps
import scipy.linalg as LA
import scipy.sparse.linalg as LAS
from fun import Fun
from newton.solver_norms import sprod


def prolong(array, Nout):
    Nin = array.shape[0]
    M = array.shape[1]
    Ndiff = Nout - Nin

    if Ndiff == 0:
        return array

    if Ndiff > 0:
        a = np.zeros((Nout, M), dtype=array.dtype, order='F')
        a[:Nin, :] = array[:]
        return a

    else:  # Ndiff < 0
        Nout = max(Nout, 0)
        return array[:Nout, :]


class LinearSolverNGSchur:
    """
        Solves the linear system obtained during Newton-Gauss continuation using
        the Schur complement trick.

        The goal is to use this class in a quasi-Gauss-Newton method.
        The steps steps are:

            1. Solve Fu(X^0) Δ = F(X^k)
            2. Solve Fu(X^0) W = F_λ(X^0)
            3. β = Vλ - Vu W
            4. δλ = - Vu W / β
            5. δu = Δ - Wu δλ

        Now at the beginning we have to update the direction vector so that
        we use the tangent at the point X^0.

        Steps to update tangent vector:
            1. β = Vλ - Vu W
            2. Tλ = - (Vu Vu + Vu W Vλ) / β
            3. Tu = Vu + W (Vλ - Tλ)

        The resulting vector Z := V^0 - T^0 is in the nullspace of F'(X^0), and
        we let t(y^0) = Z / ||Z||. The quasi-Gauss-Newton iteration guarantees that
        the corrections are in the hyperplane through y^0 with normal t(y^0).

    """
    def __init__(self, Fu, Fa, old_tangent, Projection, P=None, eps=1e-10, *args, **kwargs):
        self.eps = eps
        self.m = old_tangent.shape[1]
        self.n = Fu.shape[0] // self.m
        self.Projection = Projection

        # is the operator sparse?
        if isinstance(Fu, LAS.LinearOperator):
            Fu = Fu.to_matrix().tocsc()

        # map to the underlying type
        self.to_function = kwargs.pop('to_function', lambda x: x)
        self.to_state = kwargs.pop('to_state', lambda x: x)

        # matrices
        self.P = P
        if self.P is not None:
            self.Fu = LAS.splu(Fu * self.P)
        else:
            self.Fu = LAS.splu(Fu)

        # Compute the vector W
        self.W = self.solve_core(Fa)

        # Compute the new tangent
        self.tangent = self.new_tangent(old_tangent)

        # Compute the Schur complements
        self.schur_comp = self.tangent.a - sprod(self.tangent.u, self.W)

    def tangent_vector(self):
        return self.tangent

    def solve_core(self, rhs):
        x = self.Fu.solve(rhs)
        if self.P is not None:
            x = self.P.dot(x)

        # project
        x = self.Projection.dot(x)

        return self.to_function(x)

    def solve(self, rhs):
        # Step 1: Compute standard Newton correction.
        delta = self.solve_core(rhs)

        # Finally compute correction to u component
        delta_a = sprod(self.tangent.u, delta) / self.schur_comp
        delta_u = delta - delta_a * self.W

        return self.to_state(delta_u, delta_a)

    def new_tangent(self, tangent):
        """ Updates the known tangent by partially solving the linear system """
        # Compute Schur factor
        schur_comp = tangent.a - sprod(tangent.u, self.W)

        # Update tangent correction
        delta_t_a = - (sprod(tangent.u, tangent.u) + sprod(tangent.u, self.W * tangent.a)) / schur_comp
        delta_t_u = tangent.u + self.W * (tangent.a - delta_t_a)

        # Assemble and normalize new tangent
        dtangent = self.to_state(delta_t_u, delta_t_a)
        new_tangent = tangent - dtangent
        return new_tangent.normalize()

    def bif_pt(self, u, rvec):
        # Temp assignment
        B_u = prolong(rvec.B_u, self.n)
        B_l = rvec.B_l
        C_u = rvec.C_u
        C_l = rvec.C_l

        p_u = self.solve_core(B_u.flatten())
        self.G_l = (B_l - sprod(self.tangent.u, p_u)) / self.schur_comp
        self.G_u = p_u - self.G_l * self.W
        self.tau = -1. / (C_l * self.G_l + sprod(C_u, self.G_u))

        # Compute next value
        if rvec.C_a_u is not None:
            tau2 = -1. / (C_l * self.G_l + sprod(rvec.C_a_u, self.G_u))

        # Update the C_a_u vector
        rvec.C_a_u = p_u

        return self.tau

    def bif_dir(self):
        Y_l = - self.tau * self.G_l
        Y_u = - self.tau * self.G_u

        # Return the bifurcation direction
        return self.to_state(Y_u, Y_l)
