#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
from numpy.testing import assert_, assert_raises, assert_almost_equal
from funpy.fun import Fun


class TestFunEval:
    def test_eval(self):
        ff = lambda x : np.sin(2 * np.pi * x)
        f  = Fun(op=lambda x: np.sin(2 * np.pi * x), type='cheb')

        # test domain
        xs = np.linspace(-1, 1, 100)

        # Just check the values it returns
        assert_almost_equal(f(xs), ff(xs))

    def test_eval_domain(self):
        ff = lambda x : np.sin(2 * np.pi * x)
        f  = Fun(op=lambda x: np.sin(2 * np.pi * x), type='cheb', domain=[0, 1])

        # test domain
        xs = np.linspace(0, 1, 100)

        # Just check the values it returns
        assert_almost_equal(f(xs), ff(xs))
