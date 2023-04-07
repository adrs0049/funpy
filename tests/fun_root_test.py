#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
from numpy.testing import assert_, assert_raises, assert_almost_equal
from fun import Fun, roots


class TestFunRoots:
    def test_constant(self):
        f = Fun(op=lambda x: np.ones_like(x), domain=[-1, 1])
        r = roots(f)

        # check that roots is empty
        assert_(len(r) == 0)

    def test_constant_zero(self):
        f = Fun(op=lambda x: np.zeros_like(x), domain=[-1, 1])
        r = roots(f)

        # check that roots is empty
        assert_(len(r) == 1)
        assert_(r[0] == 0)

    def test_linear(self):
        zz = Fun(op=lambda x: x, domain=[-1, 1])
        r = roots(zz)

        # check that roots is empty
        assert_(len(r) == 1)
        assert_almost_equal(r[0], 0)

    def test_sin(self):
        f  = Fun(op=lambda x: np.sin(np.pi * x))
        r = roots(f)

        # check that roots is empty
        assert_(len(r) == 3)
        assert_almost_equal(r[1], 0)
        assert_almost_equal(r[0], -1)
        assert_almost_equal(r[2], 1)

    def test_sin2(self):
        f = Fun(op=lambda x: np.sin(2. * np.pi * x), domain=[0, 1])
        r = roots(f)

        # check that roots is empty
        assert_(len(r) == 3)
        assert_almost_equal(r[0], 0)
        assert_almost_equal(r[1], 0.5)
        assert_almost_equal(r[2], 1)
