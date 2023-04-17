#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
from numpy.testing import assert_, assert_raises, assert_almost_equal

import funpy as fp
from funpy import Fun

from colloc.chebOp import ChebOp


class TestChebOpLinear:
    def test_linear(self):
        # FIXME
        a = 10
        op     = ChebOp(functions=['u'], parameters={'a': a}, diff_order=1)
        op.eqn = ['diff(u, x) + u / (a * x**2 + 1)']
        op.bcs = ['u(-1) - 1']

        soln, success, res = op.solve(adaptive=False)

        assert_(success)
        assert_(res < 1e-9)

        # The exact solution
        e = lambda x : np.exp( - (np.arctan(np.sqrt(a) * x) + np.arctan(np.sqrt(a))) / np.sqrt(a))
        xs = np.linspace(-1, 1, 1000)
        assert_almost_equal(e(xs), soln(xs))

    def test_second_order(self):
        # Note that we don't have an exact solution for this one
        op     = ChebOp(functions=['u'], parameters={'epsilon': 0.0001})
        op.eqn = ['epsilon * diff(u, x, 2) - 2 * x * (cos(x) - 8 / 10) * diff(u, x, 1) + (cos(x) - 8 / 10) * u']
        op.bcs = ['u - 1', 'u - 1']
        g = Fun(op=lambda x: np.ones_like(x))
        soln, success, res = op.solve(g, method='nleq')

        assert_(success)
        assert_(res < 1e-8)
