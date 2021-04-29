#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
from numpy.testing import assert_, assert_raises, assert_almost_equal
from fun import Fun, norm
from colloc.chebOp import ChebOp


class TestChebOpSystem:
    def test_system_linear_us(self):
        n = 50
        op = ChebOp(functions=['u', 'v'], n=n, diff_order=1, linear=True, domain=[-np.pi, np.pi])
        op.eqn = ['u - diff(v, x)', 'diff(u, x) + v']
        op.bcs = [lambda u, v: u(-np.pi) + 1, lambda u, v: v(np.pi)]

        # guess
        soln, success, res = op.solve(adaptive=False)

        assert_(op.islinear)
        assert_(success)
        assert_(res < 1e-10)

        # The exact solution
        expected = Fun(op=[lambda x: np.cos(x), lambda x: np.sin(x)], domain=[-np.pi, np.pi])

        # compute norms
        diff_norm = norm(expected - soln)
        # TODO: figure out why this is so bad!
        assert_(diff_norm < 1e-6)

    def test_assembly_op(self):
        n = 10
        gamma = 1.5
        um = 1
        up = 3
        beta = 1
        epsilon = 1
        Du = 1e-3
        Dv = 1

        op     = ChebOp(functions=['u', 'v'], parameters={'gamma': gamma, 'u_m': um, 'u_p': up,
                                                          'beta': beta, 'epsilon': epsilon, 'Du': Du,
                                                          'Dv': Dv}, n=n)

        op.eqn = ['Du * diff(u, x, 2) + (gamma / (u_m * u_p)) * (u + u_m) * (u_p - u) * u + v',
                  'Dv * diff(v, x, 2) - epsilon * (beta * u + v)']
        op.bcs = [lambda u, v: np.diff(u)(-1), lambda u, v: np.diff(u)(1),
                  lambda u, v: np.diff(v)(-1), lambda u, v: np.diff(v)(1)]

        # verify the outputs of the various coefficients!
        g = Fun(op=[lambda x: np.cos(x * np.pi), lambda x: -0.1 * np.cos(np.pi * x)])
        g.prolong(n)

        op.discretize(u0=g)

        u = g[0]
        v = g[1]

        fu = lambda x: (gamma / (um * up)) * (2 * u(x) * up + um * up - 3 * (u**2)(x) - 2 * u(x) * um)
        Fu = Fun(op=fu)
        fv = lambda x: np.ones_like(x)
        Fv = Fun(op=fv)

        gu = lambda x: - epsilon * beta
        Gu = Fun(op=gu)
        gv = lambda x: - epsilon
        Gv = Fun(op=gv)

        xs = np.linspace(-1, 1, 1000)

        # TODO FIXME
        # first block coefficients
        # coeff = op.mat[0, 0].getCoeffs()
        # assert_almost_equal(Fu(xs), coeff[0](xs))
        # assert_almost_equal(np.zeros_like(xs), coeff[1](xs))
        # assert_almost_equal(Du*np.ones_like(xs), coeff[2](xs))

        # # first block coefficients
        # coeff = op.mat[0, 1].getCoeffs()
        # assert_almost_equal(Fv(xs), coeff[0](xs))
        # assert_almost_equal(np.zeros_like(xs), coeff[1](xs))
        # assert_almost_equal(np.zeros_like(xs), coeff[2](xs))

        # # first block coefficients
        # coeff = op.mat[1, 0].getCoeffs()
        # assert_almost_equal(Gu(xs), coeff[0](xs))
        # assert_almost_equal(np.zeros_like(xs), coeff[1](xs))
        # assert_almost_equal(np.zeros_like(xs), coeff[2](xs))

        # # first block coefficients
        # coeff = op.mat[1, 1].getCoeffs()
        # assert_almost_equal(Gv(xs), coeff[0](xs))
        # assert_almost_equal(np.zeros_like(xs), coeff[1](xs))
        # assert_almost_equal(Dv*np.ones_like(xs), coeff[2](xs))
