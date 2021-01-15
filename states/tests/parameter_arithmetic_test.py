#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
from numpy.testing import assert_, assert_raises, assert_almost_equal

from funpy.states.parameter import Parameter

class TestParameterArithmetic:
    def test_addition(self):
        p1 = Parameter(a=1)
        p2 = Parameter(a=2)
        p3 = p1 + p2

        assert_(p1.value == 1)
        assert_(p2.value == 2)
        assert_(p3.value == 3)

        assert_(p1.name == 'a')
        assert_(p2.name == 'a')
        assert_(p3.name == 'a')

    def test_addition_assign(self):
        p1 = Parameter(a=1)
        p2 = Parameter(a=2)
        p1 += p2

        assert_(p1.value == 3)
        assert_(p2.value == 2)

        assert_(p1.name == 'a')
        assert_(p2.name == 'a')

    def test_subtract(self):
        p1 = Parameter(a=1)
        p2 = Parameter(a=2)
        p3 = p1 - p2

        assert_(p1.value == 1)
        assert_(p2.value == 2)
        assert_(p3.value == -1)

        assert_(p1.name == 'a')
        assert_(p2.name == 'a')
        assert_(p3.name == 'a')

    def test_subtract_assign(self):
        p1 = Parameter(a=1)
        p2 = Parameter(a=2)
        p1 -= p2

        assert_(p1.value == -1)
        assert_(p2.value == 2)

        assert_(p1.name == 'a')
        assert_(p2.name == 'a')

    def test_mult(self):
        p1 = Parameter(a=1)
        p2 = Parameter(a=2)
        p3 = p1 * p2

        assert_(p1.value == 1)
        assert_(p2.value == 2)
        assert_(p3.value == 2)

        assert_(p1.name == 'a')
        assert_(p2.name == 'a')
        assert_(p3.name == 'a')

    def test_mul_assign(self):
        p1 = Parameter(a=1)
        p2 = Parameter(a=2)
        p1 *= p2

        assert_(p1.value == 2)
        assert_(p2.value == 2)

        assert_(p1.name == 'a')
        assert_(p2.name == 'a')

    def test_div(self):
        p1 = Parameter(a=1)
        p2 = Parameter(a=2)
        p3 = p1 / p2

        assert_(p1.value == 1)
        assert_(p2.value == 2)
        assert_(p3.value == 0.5)

        assert_(p1.name == 'a')
        assert_(p2.name == 'a')
        assert_(p3.name == 'a')

    def test_div_assign(self):
        p1 = Parameter(a=1)
        p2 = Parameter(a=2)
        p1 /= p2

        assert_(p1.value == 0.5)
        assert_(p2.value == 2)

        assert_(p1.name == 'a')
        assert_(p2.name == 'a')

    def test_ones_like(self):
        p1 = Parameter(a=2)
        p2 = np.ones_like(p1)

        assert_(p1.value == 2)
        assert_(p2.value == 1)

        assert_(p1.name == 'a')
        assert_(p2.name == 'a')

    def test_zeros_like(self):
        p1 = Parameter(a=2)
        p2 = np.zeros_like(p1)

        assert_(p1.value == 2)
        assert_(p2.value == 0)

        assert_(p1.name == 'a')
        assert_(p2.name == 'a')

    def test_inner(self):
        p1 = Parameter(a=2)
        p2 = Parameter(a=1)
        r  = np.inner(p1, p2)
        assert_(r == 2.0)

    def test_dot(self):
        p1 = Parameter(a=2)
        p2 = Parameter(a=1)
        r  = np.dot(p1, p2)
        assert_(r == 2.0)

    def test_real(self):
        p1 = Parameter(a=2 + 0j)
        p2 = np.real(p1)

        assert_(p1.value == 2 + 0j)
        assert_(p2.value == 2)

        assert_(p1.name == 'a')
        assert_(p2.name == 'a')

    def test_imag(self):
        p1 = Parameter(a=2 + 1j)
        p2 = np.imag(p1)

        assert_(p1.value == 2 + 1j)
        assert_(p2.value == 1.0)

        assert_(p1.name == 'a')
        assert_(p2.name == 'a')

    def test_sum(self):
        p1 = Parameter(a=2)
        p2 = np.sum(p1)

        assert_(p1.value == 2.0)
        assert_(p2.value == 2.0)

        assert_(p1.name == 'a')
        assert_(p2.name == 'a')
