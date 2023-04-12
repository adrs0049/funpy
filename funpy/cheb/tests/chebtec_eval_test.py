#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
from numpy.testing import assert_, assert_raises, assert_almost_equal
from cheb.chebpy import chebtec

class TestChebTecEval:
    def test_eval(self):
        ff = lambda x: np.sin(2 * np.pi * x)
        f  = chebtec(op=lambda x: np.sin(2 * np.pi * x), type='cheb')

        # test domain
        xs = np.linspace(-1, 1, 100)

        # Just check the values it returns
        assert_almost_equal(f(xs), ff(xs))

    def test_scalar_eval(self):
        ff = lambda x: np.sin(2 * np.pi * x)
        f  = chebtec(op=lambda x: np.sin(2 * np.pi * x), type='cheb')

        # test domain
        xs = np.asarray([1.0])

        # Just check the values it returns
        assert_almost_equal(f(xs), ff(xs))

    def test_real_eval(self):
        ff = lambda x: np.sin(2 * np.pi * x)
        f  = chebtec(op=lambda x: np.sin(2 * np.pi * x), type='cheb')

        # test domain
        xs = 1.0

        # Just check the values it returns
        assert_almost_equal(f(xs), ff(xs))
