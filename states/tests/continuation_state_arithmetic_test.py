#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
from numpy.testing import assert_, assert_raises, assert_almost_equal

from fun import Fun
from states.parameter import Parameter
from states.State import ContinuationState


class TestContinuationState:
    def test_addition(self):
        fun1 = Fun(op=lambda x: np.ones_like(x))
        p1 = Parameter(a=1)
        st1  = ContinuationState(u=fun1, a=p1)

        fun2 = Fun(op=lambda x: np.ones_like(x))
        p2 = Parameter(a=1)
        st2  = ContinuationState(u=fun2, a=p2)

        assert_(st1.u.values == 1)
        assert_(st2.u.values == 1)
        assert_(st1.cpar == 'a')
        assert_(st2.cpar == 'a')
        assert_(st1.a == 1)
        assert_(st2.a == 1)

        st3 = st1 + st2
        assert_(st3.u.values == 2)
        assert_(st1.u.values == 1)
        assert_(st2.u.values == 1)
        assert_(st1.cpar == 'a')
        assert_(st2.cpar == 'a')
        assert_(st3.cpar == 'a')
        assert_(st1.a == 1)
        assert_(st2.a == 1)
        assert_(st3.a == 2)

    def test_addition_nontrivial(self):
        fun1 = Fun(op=lambda x: np.cos(7 * np.pi * x / 2))
        p1 = Parameter(a=1)
        st1  = ContinuationState(u=fun1, a=p1)

        fun2 = Fun(op=lambda x: np.sin(9 * np.pi * x / 2))
        p2 = Parameter(a=1)
        st2  = ContinuationState(u=fun2, a=p2)

        xs = np.linspace(-1, 1, 1000)
        assert_almost_equal(st1.u(xs), np.cos(7 * np.pi * xs / 2))
        assert_almost_equal(st2.u(xs), np.sin(9 * np.pi * xs / 2))
        assert_(st1.cpar == 'a')
        assert_(st2.cpar == 'a')
        assert_(st1.a == 1)
        assert_(st2.a == 1)

        st3 = st1 + st2
        assert_almost_equal(st1.u(xs), np.cos(7 * np.pi * xs / 2))
        assert_almost_equal(st2.u(xs), np.sin(9 * np.pi * xs / 2))
        assert_almost_equal(st3.u(xs), np.cos(7 * np.pi * xs / 2) + np.sin(9 * np.pi * xs / 2))
        assert_(st3.cpar == 'a')
        assert_(st1.cpar == 'a')
        assert_(st2.cpar == 'a')
        assert_(st3.a == 2)
        assert_(st1.a == 1)
        assert_(st2.a == 1)

    def test_addition_assign(self):
        fun1 = Fun(op=lambda x: np.ones_like(x))
        p1 = Parameter(a=1)
        st1  = ContinuationState(u=fun1, a=p1)

        fun2 = Fun(op=lambda x: np.ones_like(x))
        p2 = Parameter(a=1)
        st2  = ContinuationState(u=fun2, a=p2)

        assert_(st1.u.values == 1)
        assert_(st2.u.values == 1)
        assert_(st1.cpar == 'a')
        assert_(st2.cpar == 'a')
        assert_(st1.a == 1)
        assert_(st2.a == 1)

        st1 += st2
        assert_(st1.u.values == 2)
        assert_(st2.u.values == 1)
        assert_(st2.cpar == 'a')
        assert_(st1.cpar == 'a')
        assert_(st1.a == 2)
        assert_(st2.a == 1)

    def test_addition_assign_nontrivial(self):
        fun1 = Fun(op=lambda x: np.cos(7 * np.pi * x / 2))
        p1 = Parameter(a=1)
        st1  = ContinuationState(u=fun1, a=p1)

        fun2 = Fun(op=lambda x: np.sin(9 * np.pi * x / 2))
        p2 = Parameter(a=1)
        st2  = ContinuationState(u=fun2, a=p2)

        xs = np.linspace(-1, 1, 1000)
        assert_almost_equal(st1.u(xs), np.cos(7 * np.pi * xs / 2))
        assert_almost_equal(st2.u(xs), np.sin(9 * np.pi * xs / 2))
        assert_(st1.cpar == 'a')
        assert_(st2.cpar == 'a')
        assert_(st1.a == 1)
        assert_(st2.a == 1)

        st1 += st2
        assert_almost_equal(st1.u(xs), np.cos(7 * np.pi * xs / 2) + np.sin(9 * np.pi * xs / 2))
        assert_almost_equal(st2.u(xs), np.sin(9 * np.pi * xs / 2))
        assert_(st2.cpar == 'a')
        assert_(st1.cpar == 'a')
        assert_(st1.a == 2)
        assert_(st2.a == 1)

    def test_subtract(self):
        fun1 = Fun(op=lambda x: np.ones_like(x))
        p1 = Parameter(a=1)
        st1  = ContinuationState(u=fun1, a=p1)

        fun2 = Fun(op=lambda x: np.ones_like(x))
        p2 = Parameter(a=1)
        st2  = ContinuationState(u=fun2, a=p2)

        assert_(st1.u.values == 1)
        assert_(st2.u.values == 1)
        assert_(st1.cpar == 'a')
        assert_(st2.cpar == 'a')
        assert_(st1.a == 1)
        assert_(st2.a == 1)

        st3 = st1 - st2
        assert_(st3.u.values == 0)
        assert_(st1.cpar == 'a')
        assert_(st2.cpar == 'a')
        assert_(st3.cpar == 'a')
        assert_(st1.a == 1)
        assert_(st2.a == 1)
        assert_(st3.a == 0)

    def test_subtract_nontrivial(self):
        fun1 = Fun(op=lambda x: np.cos(7 * np.pi * x / 2))
        p1 = Parameter(a=1)
        st1  = ContinuationState(u=fun1, a=p1)

        fun2 = Fun(op=lambda x: np.sin(9 * np.pi * x / 2))
        p2 = Parameter(a=1)
        st2  = ContinuationState(u=fun2, a=p2)

        assert_(st1.cpar == 'a')
        assert_(st2.cpar == 'a')
        assert_(st1.a == 1)
        assert_(st2.a == 1)

        xs = np.linspace(-1, 1, 1000)
        assert_almost_equal(st1.u(xs), np.cos(7 * np.pi * xs / 2))
        assert_almost_equal(st2.u(xs), np.sin(9 * np.pi * xs / 2))

        st3 = st1 - st2
        assert_almost_equal(st1.u(xs), np.cos(7 * np.pi * xs / 2))
        assert_almost_equal(st2.u(xs), np.sin(9 * np.pi * xs / 2))
        assert_almost_equal(st3.u(xs), np.cos(7 * np.pi * xs / 2) - np.sin(9 * np.pi * xs / 2))
        assert_(st3.cpar == 'a')
        assert_(st1.cpar == 'a')
        assert_(st2.cpar == 'a')
        assert_(st3.a == 0)
        assert_(st1.a == 1)
        assert_(st2.a == 1)

    def test_subtract_assign(self):
        fun1 = Fun(op=lambda x: np.ones_like(x))
        p1 = Parameter(a=1)
        st1  = ContinuationState(u=fun1, a=p1)

        fun2 = Fun(op=lambda x: np.ones_like(x))
        p2 = Parameter(a=1)
        st2  = ContinuationState(u=fun2, a=p2)

        assert_(st1.u.values == 1)
        assert_(st2.u.values == 1)
        assert_(st1.cpar == 'a')
        assert_(st2.cpar == 'a')
        assert_(st1.a == 1)
        assert_(st2.a == 1)

        st1 -= st2
        assert_(st1.u.values == 0)
        assert_(st2.u.values == 1)
        assert_(st2.cpar == 'a')
        assert_(st1.cpar == 'a')
        assert_(st1.a == 0)
        assert_(st2.a == 1)

    def test_subtract_assign_nontrivial(self):
        fun1 = Fun(op=lambda x: np.cos(7 * np.pi * x / 2))
        p1 = Parameter(a=1)
        st1  = ContinuationState(u=fun1, a=p1)

        fun2 = Fun(op=lambda x: np.sin(9 * np.pi * x / 2))
        p2 = Parameter(a=1)
        st2  = ContinuationState(u=fun2, a=p2)

        assert_(st1.cpar == 'a')
        assert_(st2.cpar == 'a')
        assert_(st1.a == 1)
        assert_(st2.a == 1)

        xs = np.linspace(-1, 1, 1000)
        assert_almost_equal(st1.u(xs), np.cos(7 * np.pi * xs / 2))
        assert_almost_equal(st2.u(xs), np.sin(9 * np.pi * xs / 2))

        st1 -= st2
        assert_almost_equal(st1.u(xs), np.cos(7 * np.pi * xs / 2) - np.sin(9 * np.pi * xs / 2))
        assert_almost_equal(st2.u(xs), np.sin(9 * np.pi * xs / 2))
        assert_(st2.cpar == 'a')
        assert_(st1.cpar == 'a')
        assert_(st1.a == 0)
        assert_(st2.a == 1)

    def test_mul(self):
        fun1 = Fun(op=lambda x: np.ones_like(x))
        p1 = Parameter(a=1)
        st1  = ContinuationState(u=fun1, a=p1)

        fun2 = Fun(op=lambda x: 2 * np.ones_like(x))
        p2 = Parameter(a=1)
        st2  = ContinuationState(u=fun2, a=p2)

        assert_(st1.u.values == 1)
        assert_(st2.u.values == 2)

        assert_(st1.cpar == 'a')
        assert_(st2.cpar == 'a')
        assert_(st1.a == 1)
        assert_(st2.a == 1)

        st3 = st1 * st2
        assert_(st1.u.values == 1)
        assert_(st2.u.values == 2)
        assert_(st3.u.values == 2)

        assert_(st3.cpar == 'a')
        assert_(st2.cpar == 'a')
        assert_(st1.cpar == 'a')
        assert_(st1.a == 1)
        assert_(st2.a == 1)
        assert_(st3.a == 1)

    def test_mul_nontrivial(self):
        fun1 = Fun(op=lambda x: np.cos(7 * np.pi * x / 2))
        p1 = Parameter(a=1)
        st1  = ContinuationState(u=fun1, a=p1)

        fun2 = Fun(op=lambda x: np.sin(9 * np.pi * x / 2))
        p2 = Parameter(a=1)
        st2  = ContinuationState(u=fun2, a=p2)

        xs = np.linspace(-1, 1, 1000)
        assert_almost_equal(st1.u(xs), np.cos(7 * np.pi * xs / 2))
        assert_almost_equal(st2.u(xs), np.sin(9 * np.pi * xs / 2))

        assert_(st1.cpar == 'a')
        assert_(st2.cpar == 'a')
        assert_(st1.a == 1)
        assert_(st2.a == 1)

        st3 = st1 * st2
        assert_almost_equal(st1.u(xs), np.cos(7 * np.pi * xs / 2))
        assert_almost_equal(st2.u(xs), np.sin(9 * np.pi * xs / 2))
        assert_almost_equal(st3.u(xs), np.cos(7 * np.pi * xs / 2) * np.sin(9 * np.pi * xs / 2))
        assert_(st3.cpar == 'a')
        assert_(st2.cpar == 'a')
        assert_(st1.cpar == 'a')
        assert_(st1.a == 1)
        assert_(st2.a == 1)
        assert_(st3.a == 1)

    def test_mul_assign(self):
        fun1 = Fun(op=lambda x: np.ones_like(x))
        p1 = Parameter(a=1)
        st1  = ContinuationState(u=fun1, a=p1)

        fun2 = Fun(op=lambda x: 2*np.ones_like(x))
        p2 = Parameter(a=2)
        st2  = ContinuationState(u=fun2, a=p2)

        assert_(st1.u.values == 1)
        assert_(st2.u.values == 2)
        assert_(st2.cpar == 'a')
        assert_(st1.cpar == 'a')
        assert_(st1.a == 1)
        assert_(st2.a == 2)

        st1 *= st2
        assert_(st1.u.values == 2)
        assert_(st2.u.values == 2)
        assert_(st2.cpar == 'a')
        assert_(st1.cpar == 'a')
        assert_(st1.a == 2)
        assert_(st2.a == 2)

    def test_mul_assign_nontrivial(self):
        fun1 = Fun(op=lambda x: np.cos(7 * np.pi * x / 2))
        p1 = Parameter(a=1)
        st1  = ContinuationState(u=fun1, a=p1)

        fun2 = Fun(op=lambda x: np.sin(9 * np.pi * x / 2))
        p2 = Parameter(a=2)
        st2  = ContinuationState(u=fun2, a=p2)

        xs = np.linspace(-1, 1, 1000)
        assert_almost_equal(st1.u(xs), np.cos(7 * np.pi * xs / 2))
        assert_almost_equal(st2.u(xs), np.sin(9 * np.pi * xs / 2))
        assert_(st2.cpar == 'a')
        assert_(st1.cpar == 'a')
        assert_(st1.a == 1)
        assert_(st2.a == 2)

        st1 *= st2
        assert_almost_equal(st1.u(xs), np.cos(7 * np.pi * xs / 2) * np.sin(9 * np.pi * xs / 2))
        assert_almost_equal(st2.u(xs), np.sin(9 * np.pi * xs / 2))
        assert_(st2.cpar == 'a')
        assert_(st1.cpar == 'a')
        assert_(st1.a == 2)
        assert_(st2.a == 2)

    def test_div(self):
        fun1 = Fun(op=lambda x: np.ones_like(x))
        p1 = Parameter(a=1)
        st1  = ContinuationState(u=fun1, a=p1)

        fun2 = Fun(op=lambda x: 2 * np.ones_like(x))
        p2 = Parameter(a=2)
        st2  = ContinuationState(u=fun2, a=p2)

        assert_(st1.u.values == 1)
        assert_(st2.u.values == 2)
        assert_(st2.cpar == 'a')
        assert_(st1.cpar == 'a')
        assert_(st1.a == 1)
        assert_(st2.a == 2)

        st3 = st1 / st2
        assert_(st1.u.values == 1)
        assert_(st2.u.values == 2)
        assert_(st3.u.values == 0.5)
        assert_(st3.cpar == 'a')
        assert_(st2.cpar == 'a')
        assert_(st1.cpar == 'a')
        assert_(st3.a == 0.5)
        assert_(st1.a == 1)
        assert_(st2.a == 2)

    def test_div_nontrivial(self):
        fun1 = Fun(op=lambda x: np.cos(7 * np.pi * x / 2))
        p1 = Parameter(a=1)
        st1  = ContinuationState(u=fun1, a=p1)

        fun2 = Fun(op=lambda x: 1.25 + np.sin(9 * np.pi * x / 2))
        p2 = Parameter(a=2)
        st2  = ContinuationState(u=fun2, a=p2)

        xs = np.linspace(-1, 1, 1000)
        assert_almost_equal(st1.u(xs), np.cos(7 * np.pi * xs / 2))
        assert_almost_equal(st2.u(xs), 1.25 + np.sin(9 * np.pi * xs / 2))
        assert_(st2.cpar == 'a')
        assert_(st1.cpar == 'a')
        assert_(st1.a == 1)
        assert_(st2.a == 2)

        st3 = st1 / st2
        assert_almost_equal(st1.u(xs), np.cos(7 * np.pi * xs / 2))
        assert_almost_equal(st2.u(xs), 1.25 + np.sin(9 * np.pi * xs / 2))
        assert_almost_equal(st3.u(xs), np.cos(7 * np.pi * xs / 2) / (1.25 + np.sin(9 * np.pi * xs / 2)))
        assert_(st3.cpar == 'a')
        assert_(st2.cpar == 'a')
        assert_(st1.cpar == 'a')
        assert_(st3.a == 0.5)
        assert_(st1.a == 1)
        assert_(st2.a == 2)

    def test_div_assign(self):
        fun1 = Fun(op=lambda x: np.ones_like(x))
        p1 = Parameter(a=1)
        st1  = ContinuationState(u=fun1, a=p1)

        fun2 = Fun(op=lambda x: 2 * np.ones_like(x))
        p2 = Parameter(a=2)
        st2  = ContinuationState(u=fun2, a=p2)

        assert_(st1.u.values == 1)
        assert_(st2.u.values == 2)
        assert_(st2.cpar == 'a')
        assert_(st1.cpar == 'a')
        assert_(st1.a == 1)
        assert_(st2.a == 2)

        st1 /= st2
        assert_(st1.u.values == 0.5)
        assert_(st2.u.values == 2)
        assert_(st2.cpar == 'a')
        assert_(st1.cpar == 'a')
        assert_(st1.a == 0.5)
        assert_(st2.a == 2)

    def test_div_assign_nontrivial(self):
        fun1 = Fun(op=lambda x: np.cos(7 * np.pi * x / 2))
        p1 = Parameter(a=1)
        st1  = ContinuationState(u=fun1, a=p1)

        fun2 = Fun(op=lambda x: 1.25 + np.sin(9 * np.pi * x / 2))
        p2 = Parameter(a=2)
        st2  = ContinuationState(u=fun2, a=p2)

        xs = np.linspace(-1, 1, 1000)
        assert_almost_equal(st1.u(xs), np.cos(7 * np.pi * xs / 2))
        assert_almost_equal(st2.u(xs), 1.25 + np.sin(9 * np.pi * xs / 2))
        assert_(st2.cpar == 'a')
        assert_(st1.cpar == 'a')
        assert_(st1.a == 1)
        assert_(st2.a == 2)

        st1 /= st2
        assert_almost_equal(st1.u(xs), np.cos(7 * np.pi * xs / 2) / (1.25 + np.sin(9 * np.pi * xs / 2)))
        assert_almost_equal(st2.u(xs), 1.25 + np.sin(9 * np.pi * xs / 2))

        assert_(st2.cpar == 'a')
        assert_(st1.cpar == 'a')
        assert_(st1.a == 0.5)
        assert_(st2.a == 2)
