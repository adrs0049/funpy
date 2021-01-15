#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
import funpy as fp
from numpy.testing import assert_, assert_raises, assert_almost_equal
from funpy.fun import Fun


class TestFun:
    def test_addition(self):
        fun1 = Fun(op=lambda x: np.ones_like(x), type='cheb')
        fun2 = Fun(op=lambda x: np.ones_like(x), type='cheb')
        fun3 = fun1 + fun2

        assert_(fun3.values == 2)
        # make sure that original functions were unchanged!
        assert_(fun1.values == 1)
        assert_(fun2.values == 1)

    def test_addition_cols(self):
        fun1 = Fun(op=[lambda x: np.ones_like(x), lambda x: np.ones_like(x)], type='cheb')
        fun2 = Fun(op=[lambda x: np.ones_like(x), lambda x: np.ones_like(x)], type='cheb')
        fun3 = fun1 + fun2

        assert_(np.all(fun3.values == 2 * np.ones((1, 1))))
        # make sure that original functions were unchanged!
        assert_(np.all(fun1.values == np.ones((1, 1))))
        assert_(np.all(fun2.values == np.ones((1, 1))))

    def test_addition_scalar(self):
        fun1 = Fun(op=lambda x: np.ones_like(x), type='cheb')
        fun2 = fun1 + 2
        fun3 = 2 + fun1

        assert_(fun2.values == 3)
        assert_(fun3.values == 3)
        # make sure that original functions were unchanged!
        assert_(fun1.values == 1)

    def test_addition_assignment(self):
        fun1 = Fun(op=lambda x: np.ones_like(x), type='cheb')
        fun2 = Fun(op=lambda x: np.ones_like(x), type='cheb')
        fun1 += fun2

        assert_(fun1.values == 2)
        # make sure that original functions were unchanged!
        assert_(fun2.values == 1)

    def test_addition_scalar_assignment(self):
        fun1 = Fun(op=lambda x: np.ones_like(x), type='cheb')
        fun1 += 2
        assert_(fun1.values == 3)

    def test_addition_functions(self):
        fun1 = Fun(op=lambda x: np.sin(2 * np.pi * x), type='cheb')
        fun2 = Fun(op=lambda x: np.cos(2 * np.pi * x), type='cheb')
        fun3 = fun1 + fun2

        xs = np.linspace(-1, 1, 1000)
        assert_almost_equal(fun3(xs), np.sin(2 * np.pi * xs) + np.cos(2 * np.pi * xs))

    def test_addition_functions2(self):
        fun1 = Fun(op=lambda x: np.sin(2 * np.pi * x), type='cheb', domain=[0, 1])
        fun2 = Fun(op=lambda x: np.cos(2 * np.pi * x), type='cheb', domain=[0, 1])
        fun3 = fun1 + fun2

        xs = np.linspace(0, 1, 1000)
        assert_almost_equal(fun3(xs), np.sin(2 * np.pi * xs) + np.cos(2 * np.pi * xs))

    def test_addition_functions3(self):
        fun1 = Fun(op=[lambda x: np.sin(2 * np.pi * x), lambda x: np.cos(2 * np.pi * x)], type='cheb')
        fun2 = Fun(op=[lambda x: np.cos(2 * np.pi * x), lambda x: np.sin(4 * np.pi * x)], type='cheb')
        fun3 = fun1 + fun2

        xs = np.linspace(-1, 1, 1000)
        assert_almost_equal(fun3(xs),
                            np.vstack((np.sin(2 * np.pi * xs) + np.cos(2 * np.pi * xs),
                                       np.cos(2 * np.pi * xs) + np.sin(4 * np.pi * xs))).T)

    def test_addition_functions4(self):
        fun1 = Fun(op=[lambda x: np.sin(2 * np.pi * x), lambda x: np.cos(2 * np.pi * x)], domain=[0,1])
        fun2 = Fun(op=[lambda x: np.cos(2 * np.pi * x), lambda x: np.sin(4 * np.pi * x)], domain=[0,1])
        fun3 = fun1 + fun2

        xs = np.linspace(0, 1, 1000)
        assert_almost_equal(fun3(xs),
                            np.vstack((np.sin(2 * np.pi * xs) + np.cos(2 * np.pi * xs),
                                       np.cos(2 * np.pi * xs) + np.sin(4 * np.pi * xs))).T)

    def test_sub(self):
        fun1 = Fun(op=lambda x: 2 * np.ones_like(x), type='cheb')
        fun2 = Fun(op=lambda x: np.ones_like(x), type='cheb')
        fun3 = fun1 - fun2

        assert_(fun3.values == 1)
        # make sure that original functions were unchanged!
        assert_(fun1.values == 2)
        assert_(fun2.values == 1)

    def test_sub_scalar(self):
        fun1 = Fun(op=lambda x: np.ones_like(x), type='cheb')
        fun2 = fun1 - 2
        fun3 = 2 - fun1

        assert_(fun2.values == -1)
        assert_(fun3.values == 1)
        # make sure that original functions were unchanged!
        assert_(fun1.values == 1)

    def test_sub_assignment(self):
        fun1 = Fun(op=lambda x: 2 * np.ones_like(x), type='cheb')
        fun2 = Fun(op=lambda x: np.ones_like(x), type='cheb')
        fun1 -= fun2

        assert_(fun1.values == 1)
        # make sure that original functions were unchanged!
        assert_(fun2.values == 1)

    def test_sub_scalar_assignment(self):
        fun1 = Fun(op=lambda x: np.ones_like(x), type='cheb')
        fun1 -= 2
        assert_(fun1.values == -1)

    def test_subtract_functions(self):
        fun1 = Fun(op=lambda x: np.sin(2 * np.pi * x), type='cheb')
        fun2 = Fun(op=lambda x: np.cos(2 * np.pi * x), type='cheb')
        fun3 = fun1 - fun2

        xs = np.linspace(-1, 1, 1000)
        assert_almost_equal(fun3(xs), np.sin(2 * np.pi * xs) - np.cos(2 * np.pi * xs))

    def test_subtract_functions2(self):
        fun1 = Fun(op=lambda x: np.sin(2 * np.pi * x), type='cheb', domain=[0, 1])
        fun2 = Fun(op=lambda x: np.cos(2 * np.pi * x), type='cheb', domain=[0, 1])
        fun3 = fun1 - fun2

        xs = np.linspace(0, 1, 1000)
        assert_almost_equal(fun3(xs), np.sin(2 * np.pi * xs) - np.cos(2 * np.pi * xs))

    def test_subtract_functions3(self):
        fun1 = Fun(op=[lambda x: np.sin(2 * np.pi * x), lambda x: np.cos(2 * np.pi * x)], type='cheb')
        fun2 = Fun(op=[lambda x: np.cos(2 * np.pi * x), lambda x: np.sin(4 * np.pi * x)], type='cheb')
        fun3 = fun1 - fun2

        xs = np.linspace(-1, 1, 1000)
        assert_almost_equal(fun3(xs),
                            np.vstack((np.sin(2 * np.pi * xs) - np.cos(2 * np.pi * xs),
                                       np.cos(2 * np.pi * xs) - np.sin(4 * np.pi * xs))).T)

    def test_subtract_functions4(self):
        fun1 = Fun(op=[lambda x: np.sin(2 * np.pi * x), lambda x: np.cos(2 * np.pi * x)], domain=[0,1])
        fun2 = Fun(op=[lambda x: np.cos(2 * np.pi * x), lambda x: np.sin(4 * np.pi * x)], domain=[0,1])
        fun3 = fun1 - fun2

        xs = np.linspace(0, 1, 1000)
        assert_almost_equal(fun3(xs),
                            np.vstack((np.sin(2 * np.pi * xs) - np.cos(2 * np.pi * xs),
                                       np.cos(2 * np.pi * xs) - np.sin(4 * np.pi * xs))).T)

    def test_mul(self):
        fun1 = Fun(op=lambda x: 2*np.ones_like(x), type='cheb')
        fun2 = Fun(op=lambda x: 2*np.ones_like(x), type='cheb')
        fun3 = fun1 * fun2

        assert_(fun3.values == 4)
        # make sure that original functions were unchanged!
        assert_(fun1.values == 2)
        assert_(fun2.values == 2)

    def test_mul_scalar(self):
        fun1 = Fun(op=lambda x: np.ones_like(x), type='cheb')
        fun2 = fun1 * 2
        fun3 = 2 * fun1

        assert_(fun2.values == 2)
        assert_(fun3.values == 2)
        # make sure that original functions were unchanged!
        assert_(fun1.values == 1)

    def test_mul_functions(self):
        fun1 = Fun(op=lambda x: np.sin(2 * np.pi * x), type='cheb')
        fun2 = Fun(op=lambda x: np.cos(2 * np.pi * x), type='cheb')
        fun3 = fun1 * fun2

        xs = np.linspace(-1, 1, 1000)
        assert_almost_equal(fun3(xs), np.sin(2 * np.pi * xs) * np.cos(2 * np.pi * xs))

    def test_mul_functions2(self):
        fun1 = Fun(op=lambda x: np.sin(2 * np.pi * x), type='cheb', domain=[0, 1])
        fun2 = Fun(op=lambda x: np.cos(2 * np.pi * x), type='cheb', domain=[0, 1])
        fun3 = fun1 * fun2

        xs = np.linspace(0, 1, 1000)
        assert_almost_equal(fun3(xs), np.sin(2 * np.pi * xs) * np.cos(2 * np.pi * xs))

    def test_mul_functions3(self):
        fun1 = Fun(op=[lambda x: np.sin(2 * np.pi * x), lambda x: np.cos(2 * np.pi * x)], type='cheb')
        fun2 = Fun(op=[lambda x: np.cos(2 * np.pi * x), lambda x: np.sin(4 * np.pi * x)], type='cheb')
        fun3 = fun1 * fun2

        xs = np.linspace(-1, 1, 1000)
        assert_almost_equal(fun3(xs),
                            np.vstack((np.sin(2 * np.pi * xs) * np.cos(2 * np.pi * xs),
                                       np.cos(2 * np.pi * xs) * np.sin(4 * np.pi * xs))).T)

    def test_mul_functions4(self):
        fun1 = Fun(op=[lambda x: np.sin(2 * np.pi * x), lambda x: np.cos(2 * np.pi * x)], domain=[0,1])
        fun2 = Fun(op=[lambda x: np.cos(2 * np.pi * x), lambda x: np.sin(4 * np.pi * x)], domain=[0,1])
        fun3 = fun1 * fun2

        xs = np.linspace(0, 1, 1000)
        assert_almost_equal(fun3(xs),
                            np.vstack((np.sin(2 * np.pi * xs) * np.cos(2 * np.pi * xs),
                                       np.cos(2 * np.pi * xs) * np.sin(4 * np.pi * xs))).T)

    def test_mul_assignment(self):
        fun1 = Fun(op=lambda x: np.ones_like(x), type='cheb')
        fun2 = Fun(op=lambda x: np.ones_like(x), type='cheb')
        fun1 += fun2

        assert_(fun1.values == 2)
        # make sure that original functions were unchanged!
        assert_(fun2.values == 1)

    def test_mul_scalar_assignment(self):
        fun1 = Fun(op=lambda x: np.ones_like(x), type='cheb')
        fun1 *= 2
        assert_(fun1.values == 2)

    def test_div(self):
        fun1 = Fun(op=lambda x: 4*np.ones_like(x), type='cheb')
        fun2 = Fun(op=lambda x: 2*np.ones_like(x), type='cheb')
        fun3 = fun1 / fun2

        assert_(fun3.values == 2)
        # make sure that original functions were unchanged!
        assert_(fun1.values == 4)
        assert_(fun2.values == 2)

    def test_div_functions(self):
        fun1 = Fun(op=lambda x: np.sin(2 * np.pi * x), type='cheb')
        fun2 = Fun(op=lambda x: 2 + np.cos(2 * np.pi * x), type='cheb')
        fun3 = fun1 / fun2

        xs = np.linspace(-1, 1, 1000)
        assert_almost_equal(fun3(xs), np.sin(2 * np.pi * xs) / (2 + np.cos(2 * np.pi * xs)))

    def test_div_functions2(self):
        fun1 = Fun(op=lambda x: np.sin(2 * np.pi * x), type='cheb', domain=[0, 1])
        fun2 = Fun(op=lambda x: 2 + np.cos(2 * np.pi * x), type='cheb', domain=[0, 1])
        fun3 = fun1 / fun2

        xs = np.linspace(0, 1, 1000)
        assert_almost_equal(fun3(xs), np.sin(2 * np.pi * xs) / (2 + np.cos(2 * np.pi * xs)))

    def test_div_assignment(self):
        fun1 = Fun(op=lambda x: 4*np.ones_like(x), type='cheb')
        fun2 = Fun(op=lambda x: 2*np.ones_like(x), type='cheb')
        fun1 /= fun2

        assert_(fun1.values == 2)
        # make sure that original functions were unchanged!
        assert_(fun2.values == 2)

    def test_div_scalar_assignment(self):
        fun1 = Fun(op=lambda x: 2*np.ones_like(x), type='cheb')
        fun1 /= 2
        assert_(fun1.values == 1)

    def test_pow(self):
        fun1 = Fun(op=lambda x: 2*np.ones_like(x), type='cheb')
        fun2 = fun1**2

        assert_(fun1.values == 2)
        assert_(fun2.values == 4)

    def test_pow_functions(self):
        fun1 = Fun(op=lambda x: np.sin(2 * np.pi * x), type='cheb')
        fun2 = Fun(op=lambda x: np.cos(2 * np.pi * x), type='cheb')
        fun3 = fun1**2
        fun4 = fun2**3

        xs = np.linspace(-1, 1, 1000)
        assert_almost_equal(fun3(xs), np.sin(2 * np.pi * xs)**2)
        assert_almost_equal(fun4(xs), np.cos(2 * np.pi * xs)**3)

    def test_pow_functions2(self):
        fun1 = Fun(op=lambda x: np.sin(2 * np.pi * x), domain=[0, 1], type='cheb')
        fun2 = Fun(op=lambda x: np.cos(2 * np.pi * x), domain=[0, 1], type='cheb')
        fun3 = fun1**2
        fun4 = fun2**3

        xs = np.linspace(0, 1, 1000)
        assert_almost_equal(fun3(xs), np.sin(2 * np.pi * xs)**2)
        assert_almost_equal(fun4(xs), np.cos(2 * np.pi * xs)**3)

    def test_pow_functions3(self):
        fun1 = Fun(op=[lambda x: np.sin(2 * np.pi * x), lambda x: np.cos(2 * np.pi * x)])
        fun3 = fun1**2

        xs = np.linspace(-1, 1, 1000)
        vals1 = np.vstack((np.sin(2 * np.pi * xs)**2, np.cos(2 * np.pi * xs)**2)).T
        assert_almost_equal(fun3(xs), vals1)

        fun2 = Fun(op=[lambda x: np.cos(2 * np.pi * x), lambda x: np.sin(4 * np.pi * x)])
        fun4 = fun2**3
        vals2 = np.vstack((np.cos(2 * np.pi * xs)**3, np.sin(4 * np.pi * xs)**3)).T
        assert_almost_equal(fun4(xs), vals2)

    def test_pow_functions4(self):
        k = 10
        fun1 = Fun(op=[lambda x: np.sin(2 * k * np.pi * x), lambda x: np.cos(2 * k * np.pi * x)])
        fun3 = fun1**2 / (2 + fun1**2)

        xs = np.linspace(-1, 1, 1000)
        f1 = np.sin(2 * k * np.pi * xs)**2
        f2 = np.cos(2 * k * np.pi * xs)**2

        vals1 = np.vstack((f1 / (2 + f1), f2 / (2 + f2)))
        assert_almost_equal(fun3(xs), vals1.T)

        fun2 = Fun(op=[lambda x: np.cos(2 * k * np.pi * x), lambda x: np.sin(4 * k * np.pi * x)])
        fun4 = fun2**3 / (2 + fun2**3)

        f1 = np.cos(2 * k * np.pi * xs)**3
        f2 = np.sin(4 * k * np.pi * xs)**3
        vals2 = np.vstack((f1 / (2 + f1), f2 / (2 + f2)))
        assert_almost_equal(fun4(xs), vals2.T)

    def test_div_power(self):
        stst = np.asarray([0.842, 1.158, 0.842, 1.158])
        f = Fun(op=[lambda x: stst[0], lambda x: stst[1],
                    lambda x: stst[2], lambda x: stst[3]], domain=[0, 1])

        h = 1.0 / (f[2]**3 + 1)
        assert_almost_equal(h.coeffs[0], 1./(0.842**3 + 1))
