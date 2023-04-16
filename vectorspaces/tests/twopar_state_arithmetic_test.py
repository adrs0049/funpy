#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
from numpy.testing import assert_, assert_raises, assert_almost_equal

from fun import Fun
from states.parameter import Parameter
from states.tp_state import TwoParameterState


class TestTwoParameterState:
    def test_addition(self):
        fun1 = Fun(op=lambda x: np.ones_like(x))
        fun2 = Fun(op=lambda x: np.ones_like(x))
        p1 = Parameter(a=1)
        p2 = Parameter(b=2)
        st1  = TwoParameterState(u=fun1, phi=fun2, p1=p1, p2=p2)

        fun3 = Fun(op=lambda x: np.ones_like(x))
        fun4 = Fun(op=lambda x: np.ones_like(x))
        p3 = Parameter(a=3)
        p4 = Parameter(b=4)
        st2  = TwoParameterState(u=fun3, phi=fun4, p1=p3, p2=p4)

        assert_(st1.u.values == 1)
        assert_(st2.u.values == 1)
        assert_(st1.phi.values == 1)
        assert_(st2.phi.values == 1)
        assert_(st1.cpar == 'b')
        assert_(st2.cpar == 'b')
        assert_(st1.a == 2)
        assert_(st2.a == 4)
        assert_(st1.b == 1)
        assert_(st2.b == 3)

        st3 = st1 + st2
        assert_(st3.u.values == 2)
        assert_(st1.u.values == 1)
        assert_(st2.u.values == 1)
        assert_(st3.phi.values == 2)
        assert_(st1.phi.values == 1)
        assert_(st2.phi.values == 1)
        assert_(st1.cpar == 'b')
        assert_(st2.cpar == 'b')
        assert_(st3.cpar == 'b')
        assert_(st1.b == 1)
        assert_(st2.b == 3)
        assert_(st3.b == 4)
        assert_(st1.a == 2)
        assert_(st2.a == 4)
        assert_(st3.a == 6)

    def test_addition_nontrivial(self):
        fun1 = Fun(op=lambda x: np.cos(7 * np.pi * x / 2))
        fun2 = Fun(op=lambda x: np.cos(5 * np.pi * x / 2))
        p1 = Parameter(a=1)
        p2 = Parameter(b=2)
        st1  = TwoParameterState(u=fun1, phi=fun2, p1=p1, p2=p2)

        fun3 = Fun(op=lambda x: np.sin(9 * np.pi * x / 2))
        fun4 = Fun(op=lambda x: np.sin(11 * np.pi * x / 2))
        p3 = Parameter(a=3)
        p4 = Parameter(b=4)
        st2  = TwoParameterState(u=fun3, phi=fun4, p1=p3, p2=p4)

        xs = np.linspace(-1, 1, 1000)
        assert_almost_equal(st1.u(xs),   np.cos(7 * np.pi * xs / 2))
        assert_almost_equal(st1.phi(xs), np.cos(5 * np.pi * xs / 2))
        assert_almost_equal(st2.u(xs),   np.sin(9 * np.pi * xs / 2))
        assert_almost_equal(st2.phi(xs), np.sin(11 * np.pi * xs / 2))
        assert_(st1.cpar == 'b')
        assert_(st2.cpar == 'b')
        assert_(st1.a == 2)
        assert_(st2.a == 4)
        assert_(st1.b == 1)
        assert_(st2.b == 3)

        st3 = st1 + st2
        assert_almost_equal(st1.u(xs),   np.cos(7 * np.pi * xs / 2))
        assert_almost_equal(st1.phi(xs), np.cos(5 * np.pi * xs / 2))
        assert_almost_equal(st2.u(xs),   np.sin(9 * np.pi * xs / 2))
        assert_almost_equal(st2.phi(xs), np.sin(11 * np.pi * xs / 2))
        assert_almost_equal(st3.u(xs), np.cos(7 * np.pi * xs / 2) + np.sin(9 * np.pi * xs / 2))
        assert_almost_equal(st3.phi(xs), np.cos(5 * np.pi * xs / 2) + np.sin(11 * np.pi * xs / 2))
        assert_(st1.cpar == 'b')
        assert_(st2.cpar == 'b')
        assert_(st3.cpar == 'b')
        assert_(st1.b == 1)
        assert_(st2.b == 3)
        assert_(st3.b == 4)
        assert_(st1.a == 2)
        assert_(st2.a == 4)
        assert_(st3.a == 6)

    def test_addition_assign(self):
        fun1 = Fun(op=lambda x: np.ones_like(x))
        fun2 = Fun(op=lambda x: np.ones_like(x))
        p1 = Parameter(a=1)
        p2 = Parameter(b=2)
        st1  = TwoParameterState(u=fun1, phi=fun2, p1=p1, p2=p2)

        fun3 = Fun(op=lambda x: np.ones_like(x))
        fun4 = Fun(op=lambda x: np.ones_like(x))
        p3 = Parameter(a=3)
        p4 = Parameter(b=4)
        st2  = TwoParameterState(u=fun3, phi=fun4, p1=p3, p2=p4)

        assert_(st1.u.values == 1)
        assert_(st2.u.values == 1)
        assert_(st1.phi.values == 1)
        assert_(st2.phi.values == 1)
        assert_(st1.cpar == 'b')
        assert_(st2.cpar == 'b')
        assert_(st1.a == 2)
        assert_(st2.a == 4)
        assert_(st1.b == 1)
        assert_(st2.b == 3)

        st1 += st2
        assert_(st1.u.values == 2)
        assert_(st2.u.values == 1)
        assert_(st1.phi.values == 2)
        assert_(st2.phi.values == 1)
        assert_(st1.cpar == 'b')
        assert_(st2.cpar == 'b')
        assert_(st1.b == 4)
        assert_(st2.b == 3)
        assert_(st1.a == 6)
        assert_(st2.a == 4)

    def test_addition_assign_nontrivial(self):
        fun1 = Fun(op=lambda x: np.cos(7 * np.pi * x / 2))
        fun2 = Fun(op=lambda x: np.cos(5 * np.pi * x / 2))
        p1 = Parameter(a=1)
        p2 = Parameter(b=2)
        st1  = TwoParameterState(u=fun1, phi=fun2, p1=p1, p2=p2)

        fun3 = Fun(op=lambda x: np.sin(9 * np.pi * x / 2))
        fun4 = Fun(op=lambda x: np.sin(11 * np.pi * x / 2))
        p3 = Parameter(a=3)
        p4 = Parameter(b=4)
        st2  = TwoParameterState(u=fun3, phi=fun4, p1=p3, p2=p4)

        xs = np.linspace(-1, 1, 1000)
        assert_almost_equal(st1.u(xs), np.cos(7 * np.pi * xs / 2))
        assert_almost_equal(st1.phi(xs), np.cos(5 * np.pi * xs / 2))
        assert_almost_equal(st2.u(xs), np.sin(9 * np.pi * xs / 2))
        assert_almost_equal(st2.phi(xs), np.sin(11 * np.pi * xs / 2))
        assert_(st1.cpar == 'b')
        assert_(st2.cpar == 'b')
        assert_(st1.a == 2)
        assert_(st2.a == 4)
        assert_(st1.b == 1)
        assert_(st2.b == 3)

        st1 += st2
        assert_almost_equal(st1.u(xs), np.cos(7 * np.pi * xs / 2) + np.sin(9 * np.pi * xs / 2))
        assert_almost_equal(st1.phi(xs), np.cos(5 * np.pi * xs / 2) + np.sin(11 * np.pi * xs / 2))
        assert_almost_equal(st2.u(xs), np.sin(9 * np.pi * xs / 2))
        assert_almost_equal(st2.phi(xs), np.sin(11 * np.pi * xs / 2))
        assert_(st1.cpar == 'b')
        assert_(st2.cpar == 'b')
        assert_(st1.b == 4)
        assert_(st2.b == 3)
        assert_(st1.a == 6)
        assert_(st2.a == 4)

    def test_subtract(self):
        fun1 = Fun(op=lambda x: np.ones_like(x))
        fun2 = Fun(op=lambda x: np.ones_like(x))
        p1 = Parameter(a=1)
        p2 = Parameter(b=2)
        st1  = TwoParameterState(u=fun1, phi=fun2, p1=p1, p2=p2)

        fun3 = Fun(op=lambda x: np.ones_like(x))
        fun4 = Fun(op=lambda x: np.ones_like(x))
        p3 = Parameter(a=3)
        p4 = Parameter(b=4)
        st2  = TwoParameterState(u=fun3, phi=fun4, p1=p3, p2=p4)

        assert_(st1.u.values == 1)
        assert_(st2.u.values == 1)
        assert_(st1.phi.values == 1)
        assert_(st2.phi.values == 1)
        assert_(st1.cpar == 'b')
        assert_(st2.cpar == 'b')
        assert_(st1.a == 2)
        assert_(st2.a == 4)
        assert_(st1.b == 1)

        st3 = st1 - st2
        assert_(st3.u.values == 0)
        assert_(st1.u.values == 1)
        assert_(st2.u.values == 1)
        assert_(st3.phi.values == 0)
        assert_(st1.phi.values == 1)
        assert_(st2.phi.values == 1)
        assert_(st1.cpar == 'b')
        assert_(st2.cpar == 'b')
        assert_(st3.cpar == 'b')
        assert_(st1.b == 1)
        assert_(st2.b == 3)
        assert_(st3.b == -2)
        assert_(st1.a == 2)
        assert_(st2.a == 4)
        assert_(st3.a == -2)

    def test_subtract_nontrivial(self):
        fun1 = Fun(op=lambda x: np.cos(7 * np.pi * x / 2))
        fun2 = Fun(op=lambda x: np.cos(5 * np.pi * x / 2))
        p1 = Parameter(a=1)
        p2 = Parameter(b=2)
        st1  = TwoParameterState(u=fun1, phi=fun2, p1=p1, p2=p2)

        fun3 = Fun(op=lambda x: np.sin(9 * np.pi * x / 2))
        fun4 = Fun(op=lambda x: np.sin(11 * np.pi * x / 2))
        p3 = Parameter(a=3)
        p4 = Parameter(b=4)
        st2  = TwoParameterState(u=fun3, phi=fun4, p1=p3, p2=p4)

        xs = np.linspace(-1, 1, 1000)
        assert_almost_equal(st1.u(xs),   np.cos(7 * np.pi * xs / 2))
        assert_almost_equal(st1.phi(xs), np.cos(5 * np.pi * xs / 2))
        assert_almost_equal(st2.u(xs),   np.sin(9 * np.pi * xs / 2))
        assert_almost_equal(st2.phi(xs), np.sin(11 * np.pi * xs / 2))
        assert_(st1.cpar == 'b')
        assert_(st2.cpar == 'b')
        assert_(st1.a == 2)
        assert_(st2.a == 4)
        assert_(st1.b == 1)
        assert_(st2.b == 3)

        st3 = st1 - st2
        assert_almost_equal(st1.u(xs),   np.cos(7 * np.pi * xs / 2))
        assert_almost_equal(st1.phi(xs), np.cos(5 * np.pi * xs / 2))
        assert_almost_equal(st2.u(xs),   np.sin(9 * np.pi * xs / 2))
        assert_almost_equal(st2.phi(xs), np.sin(11 * np.pi * xs / 2))
        assert_almost_equal(st3.u(xs),   np.cos(7 * np.pi * xs / 2) - np.sin(9 * np.pi * xs / 2))
        assert_almost_equal(st3.phi(xs), np.cos(5 * np.pi * xs / 2) - np.sin(11 * np.pi * xs / 2))
        assert_(st1.cpar == 'b')
        assert_(st2.cpar == 'b')
        assert_(st3.cpar == 'b')
        assert_(st1.b == 1)
        assert_(st2.b == 3)
        assert_(st3.b == -2)
        assert_(st1.a == 2)
        assert_(st2.a == 4)
        assert_(st3.a == -2)

    def test_subtract_assign(self):
        fun1 = Fun(op=lambda x: np.ones_like(x))
        fun2 = Fun(op=lambda x: np.ones_like(x))
        p1 = Parameter(a=1)
        p2 = Parameter(b=2)
        st1  = TwoParameterState(u=fun1, phi=fun2, p1=p1, p2=p2)

        fun3 = Fun(op=lambda x: np.ones_like(x))
        fun4 = Fun(op=lambda x: np.ones_like(x))
        p3 = Parameter(a=3)
        p4 = Parameter(b=4)
        st2  = TwoParameterState(u=fun3, phi=fun4, p1=p3, p2=p4)

        assert_(st1.u.values == 1)
        assert_(st2.u.values == 1)
        assert_(st1.phi.values == 1)
        assert_(st2.phi.values == 1)
        assert_(st1.cpar == 'b')
        assert_(st2.cpar == 'b')
        assert_(st1.a == 2)
        assert_(st2.a == 4)
        assert_(st1.b == 1)
        assert_(st2.b == 3)

        st1 -= st2
        assert_(st1.u.values == 0)
        assert_(st2.u.values == 1)
        assert_(st1.phi.values == 0)
        assert_(st2.phi.values == 1)
        assert_(st1.cpar == 'b')
        assert_(st2.cpar == 'b')
        assert_(st1.b == -2)
        assert_(st2.b == 3)
        assert_(st1.a == -2)
        assert_(st2.a == 4)

    def test_subtract_assign_nontrivial(self):
        fun1 = Fun(op=lambda x: np.cos(7 * np.pi * x / 2))
        fun2 = Fun(op=lambda x: np.cos(5 * np.pi * x / 2))
        p1 = Parameter(a=1)
        p2 = Parameter(b=2)
        st1  = TwoParameterState(u=fun1, phi=fun2, p1=p1, p2=p2)

        fun3 = Fun(op=lambda x: np.sin(9 * np.pi * x / 2))
        fun4 = Fun(op=lambda x: np.sin(11 * np.pi * x / 2))
        p3 = Parameter(a=3)
        p4 = Parameter(b=4)
        st2  = TwoParameterState(u=fun3, phi=fun4, p1=p3, p2=p4)

        xs = np.linspace(-1, 1, 1000)
        assert_almost_equal(st1.u(xs), np.cos(7 * np.pi * xs / 2))
        assert_almost_equal(st1.phi(xs), np.cos(5 * np.pi * xs / 2))
        assert_almost_equal(st2.u(xs), np.sin(9 * np.pi * xs / 2))
        assert_almost_equal(st2.phi(xs), np.sin(11 * np.pi * xs / 2))
        assert_(st1.cpar == 'b')
        assert_(st2.cpar == 'b')
        assert_(st1.a == 2)
        assert_(st2.a == 4)
        assert_(st1.b == 1)
        assert_(st2.b == 3)

        st1 -= st2
        assert_almost_equal(st1.u(xs), np.cos(7 * np.pi * xs / 2) - np.sin(9 * np.pi * xs / 2))
        assert_almost_equal(st1.phi(xs), np.cos(5 * np.pi * xs / 2) - np.sin(11 * np.pi * xs / 2))
        assert_almost_equal(st2.u(xs), np.sin(9 * np.pi * xs / 2))
        assert_almost_equal(st2.phi(xs), np.sin(11 * np.pi * xs / 2))
        assert_(st1.cpar == 'b')
        assert_(st2.cpar == 'b')
        assert_(st1.b == -2)
        assert_(st2.b == 3)
        assert_(st1.a == -2)
        assert_(st2.a == 4)

    def test_mul(self):
        fun1 = Fun(op=lambda x: np.ones_like(x))
        fun2 = Fun(op=lambda x: np.ones_like(x))
        p1 = Parameter(a=1)
        p2 = Parameter(b=2)
        st1  = TwoParameterState(u=fun1, phi=fun2, p1=p1, p2=p2)

        fun3 = Fun(op=lambda x: 2 * np.ones_like(x))
        fun4 = Fun(op=lambda x: 3 * np.ones_like(x))
        p3 = Parameter(a=3)
        p4 = Parameter(b=4)
        st2  = TwoParameterState(u=fun3, phi=fun4, p1=p3, p2=p4)

        assert_(st1.u.values == 1)
        assert_(st2.u.values == 2)
        assert_(st1.phi.values == 1)
        assert_(st2.phi.values == 3)
        assert_(st1.cpar == 'b')
        assert_(st2.cpar == 'b')
        assert_(st1.a == 2)
        assert_(st2.a == 4)
        assert_(st1.b == 1)
        assert_(st2.b == 3)

        st3 = st1 * st2
        assert_(st3.u.values == 2)
        assert_(st1.u.values == 1)
        assert_(st2.u.values == 2)
        assert_(st3.phi.values == 3)
        assert_(st1.phi.values == 1)
        assert_(st2.phi.values == 3)
        assert_(st1.cpar == 'b')
        assert_(st2.cpar == 'b')
        assert_(st3.cpar == 'b')
        assert_(st1.b == 1)
        assert_(st2.b == 3)
        assert_(st3.b == 3)
        assert_(st1.a == 2)
        assert_(st2.a == 4)
        assert_(st3.a == 8)

    def test_mul_nontrivial(self):
        fun1 = Fun(op=lambda x: np.cos(7 * np.pi * x / 2))
        fun2 = Fun(op=lambda x: np.cos(5 * np.pi * x / 2))
        p1 = Parameter(a=1)
        p2 = Parameter(b=2)
        st1  = TwoParameterState(u=fun1, phi=fun2, p1=p1, p2=p2)

        fun3 = Fun(op=lambda x: np.sin(9 * np.pi * x / 2))
        fun4 = Fun(op=lambda x: np.sin(11 * np.pi * x / 2))
        p3 = Parameter(a=3)
        p4 = Parameter(b=4)
        st2  = TwoParameterState(u=fun3, phi=fun4, p1=p3, p2=p4)

        xs = np.linspace(-1, 1, 1000)
        assert_almost_equal(st1.u(xs),   np.cos(7 * np.pi * xs / 2))
        assert_almost_equal(st1.phi(xs), np.cos(5 * np.pi * xs / 2))
        assert_almost_equal(st2.u(xs),   np.sin(9 * np.pi * xs / 2))
        assert_almost_equal(st2.phi(xs), np.sin(11 * np.pi * xs / 2))
        assert_(st1.cpar == 'b')
        assert_(st2.cpar == 'b')
        assert_(st1.a == 2)
        assert_(st2.a == 4)
        assert_(st1.b == 1)
        assert_(st2.b == 3)

        st3 = st1 * st2
        assert_almost_equal(st1.u(xs),   np.cos(7 * np.pi * xs / 2))
        assert_almost_equal(st1.phi(xs), np.cos(5 * np.pi * xs / 2))
        assert_almost_equal(st2.u(xs),   np.sin(9 * np.pi * xs / 2))
        assert_almost_equal(st2.phi(xs), np.sin(11 * np.pi * xs / 2))
        assert_almost_equal(st3.u(xs),   np.cos(7 * np.pi * xs / 2) * np.sin(9 * np.pi * xs / 2))
        assert_almost_equal(st3.phi(xs), np.cos(5 * np.pi * xs / 2) * np.sin(11 * np.pi * xs / 2))
        assert_(st1.cpar == 'b')
        assert_(st2.cpar == 'b')
        assert_(st3.cpar == 'b')
        assert_(st1.b == 1)
        assert_(st2.b == 3)
        assert_almost_equal(st3.b, 3)
        assert_(st1.a == 2)
        assert_(st2.a == 4)
        assert_almost_equal(st3.a, 8)

    def test_mul_assign(self):
        fun1 = Fun(op=lambda x: np.ones_like(x))
        fun2 = Fun(op=lambda x: np.ones_like(x))
        p1 = Parameter(a=1)
        p2 = Parameter(b=2)
        st1  = TwoParameterState(u=fun1, phi=fun2, p1=p1, p2=p2)

        fun3 = Fun(op=lambda x: 2 * np.ones_like(x))
        fun4 = Fun(op=lambda x: 3 * np.ones_like(x))
        p3 = Parameter(a=3)
        p4 = Parameter(b=4)
        st2  = TwoParameterState(u=fun3, phi=fun4, p1=p3, p2=p4)

        assert_(st1.u.values == 1)
        assert_(st2.u.values == 2)
        assert_(st1.phi.values == 1)
        assert_(st2.phi.values == 3)
        assert_(st1.cpar == 'b')
        assert_(st2.cpar == 'b')
        assert_(st1.a == 2)
        assert_(st2.a == 4)
        assert_(st1.b == 1)
        assert_(st2.b == 3)

        st1 *= st2
        assert_(st1.u.values == 2)
        assert_(st2.u.values == 2)
        assert_(st1.phi.values == 3)
        assert_(st2.phi.values == 3)
        assert_(st1.cpar == 'b')
        assert_(st2.cpar == 'b')
        assert_(st1.b == 3)
        assert_(st2.b == 3)
        assert_(st1.a == 8)
        assert_(st2.a == 4)

    def test_mul_assign_nontrivial(self):
        fun1 = Fun(op=lambda x: np.cos(7 * np.pi * x / 2))
        fun2 = Fun(op=lambda x: np.cos(5 * np.pi * x / 2))
        p1 = Parameter(a=1)
        p2 = Parameter(b=2)
        st1  = TwoParameterState(u=fun1, phi=fun2, p1=p1, p2=p2)

        fun3 = Fun(op=lambda x: np.sin(9 * np.pi * x / 2))
        fun4 = Fun(op=lambda x: np.sin(11 * np.pi * x / 2))
        p3 = Parameter(a=3)
        p4 = Parameter(b=4)
        st2  = TwoParameterState(u=fun3, phi=fun4, p1=p3, p2=p4)

        xs = np.linspace(-1, 1, 1000)
        assert_almost_equal(st1.u(xs), np.cos(7 * np.pi * xs / 2))
        assert_almost_equal(st1.phi(xs), np.cos(5 * np.pi * xs / 2))
        assert_almost_equal(st2.u(xs), np.sin(9 * np.pi * xs / 2))
        assert_almost_equal(st2.phi(xs), np.sin(11 * np.pi * xs / 2))
        assert_(st1.cpar == 'b')
        assert_(st2.cpar == 'b')
        assert_(st1.a == 2)
        assert_(st2.a == 4)
        assert_(st1.b == 1)
        assert_(st2.b == 3)

        st1 *= st2
        assert_almost_equal(st1.u(xs), np.cos(7 * np.pi * xs / 2) * np.sin(9 * np.pi * xs / 2))
        assert_almost_equal(st1.phi(xs), np.cos(5 * np.pi * xs / 2) * np.sin(11 * np.pi * xs / 2))
        assert_almost_equal(st2.u(xs), np.sin(9 * np.pi * xs / 2))
        assert_almost_equal(st2.phi(xs), np.sin(11 * np.pi * xs / 2))
        assert_(st1.cpar == 'b')
        assert_(st2.cpar == 'b')
        assert_(st1.b == 3)
        assert_(st2.b == 3)
        assert_(st1.a == 8)
        assert_(st2.a == 4)

    def test_div(self):
        fun1 = Fun(op=lambda x: np.ones_like(x))
        fun2 = Fun(op=lambda x: np.ones_like(x))
        p1 = Parameter(a=1)
        p2 = Parameter(b=2)
        st1  = TwoParameterState(u=fun1, phi=fun2, p1=p1, p2=p2)

        fun3 = Fun(op=lambda x: 2 * np.ones_like(x))
        fun4 = Fun(op=lambda x: 3 * np.ones_like(x))
        p3 = Parameter(a=3)
        p4 = Parameter(b=4)
        st2  = TwoParameterState(u=fun3, phi=fun4, p1=p3, p2=p4)

        assert_(st1.u.values == 1)
        assert_(st2.u.values == 2)
        assert_(st1.phi.values == 1)
        assert_(st2.phi.values == 3)
        assert_(st1.cpar == 'b')
        assert_(st2.cpar == 'b')
        assert_(st1.a == 2)
        assert_(st2.a == 4)
        assert_(st1.b == 1)
        assert_(st2.b == 3)

        st3 = st1 / st2
        assert_almost_equal(st3.u.values, 0.5)
        assert_(st1.u.values == 1)
        assert_(st2.u.values == 2)
        assert_almost_equal(st3.phi.values, 1./3)
        assert_(st1.phi.values == 1)
        assert_(st2.phi.values == 3)
        assert_(st1.cpar == 'b')
        assert_(st2.cpar == 'b')
        assert_(st3.cpar == 'b')
        assert_(st1.b == 1)
        assert_(st2.b == 3)
        assert_almost_equal(st3.b, 1./3)
        assert_(st1.a == 2)
        assert_(st2.a == 4)
        assert_almost_equal(st3.a, 0.5)

    def test_div_nontrivial(self):
        fun1 = Fun(op=lambda x: np.cos(7 * np.pi * x / 2))
        fun2 = Fun(op=lambda x: np.cos(5 * np.pi * x / 2))
        p1 = Parameter(a=1)
        p2 = Parameter(b=2)
        st1  = TwoParameterState(u=fun1, phi=fun2, p1=p1, p2=p2)

        fun3 = Fun(op=lambda x: 1.25 + np.sin(9 * np.pi * x / 2))
        fun4 = Fun(op=lambda x: 1.25 + np.sin(11 * np.pi * x / 2))
        p3 = Parameter(a=3)
        p4 = Parameter(b=4)
        st2  = TwoParameterState(u=fun3, phi=fun4, p1=p3, p2=p4)

        xs = np.linspace(-1, 1, 1000)
        assert_almost_equal(st1.u(xs),   np.cos(7 * np.pi * xs / 2))
        assert_almost_equal(st1.phi(xs), np.cos(5 * np.pi * xs / 2))
        assert_almost_equal(st2.u(xs),   1.25 + np.sin(9 * np.pi * xs / 2))
        assert_almost_equal(st2.phi(xs), 1.25 + np.sin(11 * np.pi * xs / 2))
        assert_(st1.cpar == 'b')
        assert_(st2.cpar == 'b')
        assert_(st1.a == 2)
        assert_(st2.a == 4)
        assert_(st1.b == 1)
        assert_(st2.b == 3)

        st3 = st1 / st2
        assert_almost_equal(st1.u(xs),   np.cos(7 * np.pi * xs / 2))
        assert_almost_equal(st1.phi(xs), np.cos(5 * np.pi * xs / 2))
        assert_almost_equal(st2.u(xs),   1.25 + np.sin(9 * np.pi * xs / 2))
        assert_almost_equal(st2.phi(xs), 1.25 + np.sin(11 * np.pi * xs / 2))
        assert_almost_equal(st3.u(xs),   np.cos(7 * np.pi * xs / 2) / (1.25 +  np.sin(9 * np.pi * xs / 2)))
        assert_almost_equal(st3.phi(xs), np.cos(5 * np.pi * xs / 2) / (1.25 +  np.sin(11 * np.pi * xs / 2)))
        assert_(st1.cpar == 'b')
        assert_(st2.cpar == 'b')
        assert_(st3.cpar == 'b')
        assert_(st1.b == 1)
        assert_(st2.b == 3)
        assert_almost_equal(st3.b, 1./3)
        assert_(st1.a == 2)
        assert_(st2.a == 4)
        assert_almost_equal(st3.a, 0.5)

    def test_div_assign(self):
        fun1 = Fun(op=lambda x: np.ones_like(x))
        fun2 = Fun(op=lambda x: np.ones_like(x))
        p1 = Parameter(a=1)
        p2 = Parameter(b=2)
        st1  = TwoParameterState(u=fun1, phi=fun2, p1=p1, p2=p2)

        fun3 = Fun(op=lambda x: 2 * np.ones_like(x))
        fun4 = Fun(op=lambda x: 3 * np.ones_like(x))
        p3 = Parameter(a=3)
        p4 = Parameter(b=4)
        st2  = TwoParameterState(u=fun3, phi=fun4, p1=p3, p2=p4)

        assert_(st1.u.values == 1)
        assert_(st2.u.values == 2)
        assert_(st1.phi.values == 1)
        assert_(st2.phi.values == 3)
        assert_(st1.cpar == 'b')
        assert_(st2.cpar == 'b')
        assert_(st1.a == 2)
        assert_(st2.a == 4)
        assert_(st1.b == 1)
        assert_(st2.b == 3)

        st1 /= st2
        assert_almost_equal(st1.u.values, 0.5)
        assert_(st2.u.values == 2)
        assert_almost_equal(st1.phi.values, 1./3)
        assert_(st2.phi.values == 3)
        assert_(st1.cpar == 'b')
        assert_(st2.cpar == 'b')
        assert_almost_equal(st1.b, 1./3)
        assert_(st2.b == 3)
        assert_almost_equal(st1.a, 0.5)
        assert_(st2.a == 4)

    def test_div_assign_nontrivial(self):
        fun1 = Fun(op=lambda x: np.cos(7 * np.pi * x / 2))
        fun2 = Fun(op=lambda x: np.cos(5 * np.pi * x / 2))
        p1 = Parameter(a=1)
        p2 = Parameter(b=2)
        st1  = TwoParameterState(u=fun1, phi=fun2, p1=p1, p2=p2)

        fun3 = Fun(op=lambda x: 1.25 + np.sin(9 * np.pi * x / 2))
        fun4 = Fun(op=lambda x: 1.25 + np.sin(11 * np.pi * x / 2))
        p3 = Parameter(a=3)
        p4 = Parameter(b=4)
        st2  = TwoParameterState(u=fun3, phi=fun4, p1=p3, p2=p4)

        xs = np.linspace(-1, 1, 1000)
        assert_almost_equal(st1.u(xs),   np.cos(7 * np.pi * xs / 2))
        assert_almost_equal(st1.phi(xs), np.cos(5 * np.pi * xs / 2))
        assert_almost_equal(st2.u(xs),   1.25 + np.sin(9 * np.pi * xs / 2))
        assert_almost_equal(st2.phi(xs), 1.25 + np.sin(11 * np.pi * xs / 2))
        assert_(st1.cpar == 'b')
        assert_(st2.cpar == 'b')
        assert_(st1.a == 2)
        assert_(st2.a == 4)
        assert_(st1.b == 1)
        assert_(st2.b == 3)

        st1 /= st2
        assert_almost_equal(st1.u(xs), np.cos(7 * np.pi * xs / 2) / (1.25 + np.sin(9 * np.pi * xs / 2)))
        assert_almost_equal(st1.phi(xs), np.cos(5 * np.pi * xs / 2) / (1.25 + np.sin(11 * np.pi * xs / 2)))
        assert_almost_equal(st2.u(xs), 1.25 + np.sin(9 * np.pi * xs / 2))
        assert_almost_equal(st2.phi(xs), 1.25 + np.sin(11 * np.pi * xs / 2))
        assert_(st1.cpar == 'b')
        assert_(st2.cpar == 'b')
        assert_almost_equal(st1.b, 1./3)
        assert_(st2.b == 3)
        assert_almost_equal(st1.a, 0.5)
        assert_(st2.a == 4)