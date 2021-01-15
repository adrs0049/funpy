#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
from numpy.testing import assert_, assert_raises, assert_almost_equal
from copy import copy, deepcopy
from funpy.fun import Fun


class TestFunEval:
    def test_copy(self):
        ff = lambda x : np.sin(2 * np.pi * x)
        f  = Fun(op=lambda x: np.sin(2 * np.pi * x), type='cheb')

        cf = copy(f)
        # TODO: finish this test!

        # test domain
        xs = np.linspace(0, 1, 100)

        # Just check the values it returns
        assert_almost_equal(f(xs), ff(xs))

    def test_deepcopy(self):
        ff = lambda x : np.sin(2 * np.pi * x)
        f  = Fun(op=lambda x: np.sin(2 * np.pi * x), type='cheb', domain=[0, 1])

        # copy the function f
        df = deepcopy(f)
        # TODO: FIXME this should not be required but I guess simplify gets called somewhere!
        df = df.prolong(f.n)

        # modify f
        f.values = np.ones(f.n)
        f.coeffs = np.ones(f.n)

        # Check that df and f are different
        assert_(np.all(np.abs(f.coeffs.flatten() - df.coeffs.flatten()) > 1e-4))

    def test_zeros_like(self):
        f = Fun(op=lambda x: np.sin(2 * np.pi * x), type='cheb', domain=[-1, 1])
        g = np.zeros_like(f)
        assert_(np.all(g.coeffs == np.zeros_like(f.coeffs)))

    def test_ones_like(self):
        f = Fun(op=lambda x: np.sin(2 * np.pi * x), type='cheb', domain=[-1, 1])
        g = np.ones_like(f)
        assert_almost_equal(g.coeffs, np.expand_dims(np.hstack((1, np.zeros(f.n-1))), axis=1))
