#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
from numpy.testing import assert_, assert_raises, assert_almost_equal
from cheb.chebpy import chebtec

class TestChebTec:
    def test_addition(self):
        fun1 = chebtec(op=lambda x: np.ones_like(x), type='cheb')
        fun2 = chebtec(op=lambda x: np.ones_like(x), type='cheb')
        fun3 = fun1 + fun2

        assert_(fun3.values == 2)
        assert_(fun3.coeffs == 2)
        # make sure that original functions were unchanged!
        assert_(fun1.values == 1)
        assert_(fun2.values == 1)
        assert_(fun1.coeffs == 1)
        assert_(fun2.coeffs == 1)

    def test_addition_functions(self):
        fun1 = chebtec(op=lambda x: np.sin(2 * np.pi * x), type='cheb')
        fun2 = chebtec(op=lambda x: np.cos(2 * np.pi * x), type='cheb')
        fun3 = fun1 + fun2

        xs = np.linspace(-1, 1, 1000)
        assert_almost_equal(fun3(xs), np.sin(2 * np.pi * xs) + np.cos(2 * np.pi * xs))

    def test_addition_functions2(self):
        fun1 = chebtec(op=lambda x: x**2, type='cheb')
        fun2 = chebtec(op=lambda x: np.cos(2 * np.pi * x), type='cheb')
        fun3 = fun1 + fun2

        xs = np.linspace(-1, 1, 1000)
        assert_almost_equal(fun3(xs), xs**2 + np.cos(2 * np.pi * xs))

    def test_addition_scalar(self):
        fun1 = chebtec(op=lambda x: np.ones_like(x), type='cheb')
        fun2 = fun1 + 2
        assert_(fun1.values == 1)
        assert_(fun1.coeffs == 1)
        assert_(fun2.values == 3)
        assert_(fun2.coeffs == 3)

        fun3 = 2 + fun1
        assert_(fun3.values == 3)
        assert_(fun3.coeffs == 3)
        assert_(fun1.values == 1)
        assert_(fun1.coeffs == 1)

    def test_addition_assignment(self):
        fun1 = chebtec(op=lambda x: np.ones_like(x), type='cheb')
        fun2 = chebtec(op=lambda x: np.ones_like(x), type='cheb')
        fun1 += fun2

        assert_(fun1.values == 2)
        assert_(fun1.coeffs == 2)
        # make sure that original functions were unchanged!
        assert_(fun2.values == 1)
        assert_(fun2.coeffs == 1)

    def test_addition_scalar_assignment(self):
        fun1 = chebtec(op=lambda x: np.ones_like(x), type='cheb')
        fun1 += 2
        assert_(fun1.values == 3)
        assert_(fun1.coeffs == 3)

    def test_negative(self):
        fun1 = chebtec(op=lambda x: np.ones_like(x), type='cheb')
        fun2 = -fun1
        assert_(fun1.values == 1)
        assert_(fun1.coeffs == 1)
        assert_(fun2.values == -1)
        assert_(fun2.coeffs == -1)

    def test_positive(self):
        fun1 = chebtec(op=lambda x: np.ones_like(x), type='cheb')
        fun2 = +fun1
        assert_(fun1.values == 1)
        assert_(fun1.coeffs == 1)
        assert_(fun2.values == 1)
        assert_(fun2.coeffs == 1)

    def test_sub(self):
        fun1 = chebtec(op=lambda x: 2 * np.ones_like(x), type='cheb')
        fun2 = chebtec(op=lambda x: np.ones_like(x), type='cheb')
        fun3 = fun1 - fun2

        assert_(fun3.values == 1)
        assert_(fun3.coeffs == 1)
        # make sure that original functions were unchanged!
        assert_(fun1.values == 2)
        assert_(fun2.values == 1)
        assert_(fun1.coeffs == 2)
        assert_(fun2.coeffs == 1)

    def test_subtract_functions(self):
        fun1 = chebtec(op=lambda x: np.sin(2 * np.pi * x), type='cheb')
        fun2 = chebtec(op=lambda x: np.cos(2 * np.pi * x), type='cheb')
        fun3 = fun1 - fun2

        xs = np.linspace(-1, 1, 1000)
        assert_almost_equal(fun3(xs), np.sin(2 * np.pi * xs) - np.cos(2 * np.pi * xs))

    def test_sub_scalar(self):
        fun1 = chebtec(op=lambda x: np.ones_like(x), type='cheb')
        fun2 = fun1 - 2
        fun3 = 2 - fun1

        assert_(fun2.values == -1)
        assert_(fun2.coeffs == -1)
        assert_(fun3.values == 1)
        assert_(fun3.coeffs == 1)
        # make sure that original functions were unchanged!
        assert_(fun1.values == 1)
        assert_(fun1.coeffs == 1)

    def test_sub_assignment(self):
        fun1 = chebtec(op=lambda x: 2 * np.ones_like(x), type='cheb')
        fun2 = chebtec(op=lambda x: np.ones_like(x), type='cheb')
        fun1 -= fun2

        assert_(fun1.values == 1)
        assert_(fun1.coeffs == 1)
        # make sure that original functions were unchanged!
        assert_(fun2.values == 1)
        assert_(fun2.coeffs == 1)

    def test_sub_scalar_assignment(self):
        fun1 = chebtec(op=lambda x: np.ones_like(x), type='cheb')
        fun1 -= 2
        assert_(fun1.coeffs == -1)

    def test_mul(self):
        fun1 = chebtec(op=lambda x: 2*np.ones_like(x), type='cheb')
        fun2 = chebtec(op=lambda x: 2*np.ones_like(x), type='cheb')
        fun3 = fun1 * fun2

        assert_(fun3.values == 4)
        assert_(fun3.coeffs == 4)
        # make sure that original functions were unchanged!
        assert_(fun1.values == 2)
        assert_(fun2.values == 2)
        assert_(fun1.coeffs == 2)
        assert_(fun2.coeffs == 2)

    def test_mul_scalar(self):
        fun1 = chebtec(op=lambda x: np.ones_like(x), type='cheb')
        fun2 = fun1 * 2
        fun3 = 2 * fun1

        assert_(fun2.values == 2)
        assert_(fun3.values == 2)
        assert_(fun2.coeffs == 2)
        assert_(fun3.coeffs == 2)
        # make sure that original functions were unchanged!
        assert_(fun1.values == 1)
        assert_(fun1.coeffs == 1)

    def test_mul_assignment(self):
        fun1 = chebtec(op=lambda x: np.ones_like(x), type='cheb')
        fun2 = chebtec(op=lambda x: np.ones_like(x), type='cheb')
        fun1 += fun2

        assert_(fun1.values == 2)
        assert_(fun1.coeffs == 2)
        # make sure that original functions were unchanged!
        assert_(fun2.values == 1)
        assert_(fun2.coeffs == 1)

    def test_mul_scalar_assignment(self):
        fun1 = chebtec(op=lambda x: np.ones_like(x), type='cheb')
        fun1 *= 2
        #assert_(fun1.values == 2)
        assert_(fun1.coeffs == 2)

    def test_div(self):
        fun1 = chebtec(op=lambda x: 4*np.ones_like(x), type='cheb')
        fun2 = chebtec(op=lambda x: 2*np.ones_like(x), type='cheb')
        fun3 = fun1 / fun2

        assert_(fun3.values == 2)
        assert_(fun3.coeffs == 2)
        # make sure that original functions were unchanged!
        assert_(fun1.values == 4)
        assert_(fun2.values == 2)
        assert_(fun1.coeffs == 4)
        assert_(fun2.coeffs == 2)

    def test_div_assignment(self):
        fun1 = chebtec(op=lambda x: 4*np.ones_like(x), type='cheb')
        fun2 = chebtec(op=lambda x: 2*np.ones_like(x), type='cheb')
        fun1 /= fun2

        assert_(fun1.values == 2)
        assert_(fun1.coeffs == 2)
        # make sure that original functions were unchanged!
        assert_(fun2.values == 2)
        assert_(fun2.coeffs == 2)

    def test_div_scalar_assignment(self):
        fun1 = chebtec(op=lambda x: 2*np.ones_like(x), type='cheb')
        fun1 /= 2
        assert_(fun1.values == 1)
        assert_(fun1.coeffs == 1)

    def test_sin_scalar(self):
        fun1 = chebtec(op=lambda x: np.ones_like(x), type='cheb')
        fun2 = np.sin(fun1)
        assert_(fun1.values == 1)
        assert_(fun1.coeffs == 1)
        assert_(fun2.values == np.sin(1))
        assert_(fun2.coeffs == np.sin(1))

    def test_pow(self):
        fun1 = chebtec(op=lambda x: 2*np.ones_like(x), type='cheb')
        fun2 = fun1**2
        assert_(fun1.values == 2)
        assert_(fun1.coeffs == 2)
        assert_(fun2.values == 4)
        assert_(fun2.coeffs == 4)

    def test_pow2(self):
        fun1 = chebtec(op=lambda x: np.cos(2 * np.pi * x), type='cheb')
        fun2 = fun1**2
        xs = np.linspace(-1, 1, 1000)
        assert_almost_equal(fun1(xs), np.cos(2 * np.pi * xs))
        assert_almost_equal(fun2(xs), np.cos(2 * np.pi * xs)**2)
