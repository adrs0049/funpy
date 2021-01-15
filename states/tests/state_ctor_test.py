#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
from numpy.testing import assert_, assert_raises, assert_almost_equal

from funpy.fun import Fun
from funpy.states.deflation_state import DeflationState
from funpy.states.State import ContinuationState
from funpy.states.tp_state import TwoParameterState
from funpy.states.parameter import Parameter


class TestConstructionState:
    def test_zeros_deflation(self):
        fun1 = Fun(op=lambda x: np.ones_like(x))
        st1  = DeflationState(u=fun1)

        nst = np.zeros_like(st1)

        assert_(st1.u.values == 1)
        assert_(nst.u.values == 0)

    def test_ones_deflation(self):
        fun1 = Fun(op=lambda x: 2 * np.ones_like(x))
        st1  = DeflationState(u=fun1)

        nst = np.ones_like(st1)

        assert_(st1.u.values == 2)
        assert_(nst.u.values == 1)

    def test_zeros_cont(self):
        fun1 = Fun(op=lambda x: np.ones_like(x))
        p1 = Parameter(gamma=2)
        st1  = ContinuationState(u=fun1, a=p1)

        nst = np.zeros_like(st1)

        assert_(st1.u.values == 1)
        assert_(st1.a == 2)

        assert_(nst.u.values == 0)
        assert_(nst.a == 0)

    def test_ones_cont(self):
        fun1 = Fun(op=lambda x: 2 * np.ones_like(x))
        p1 = Parameter(gamma=2)
        st1  = ContinuationState(u=fun1, a=p1)

        nst = np.ones_like(st1)

        assert_(st1.u.values == 2)
        assert_(st1.a == 2)

        assert_(nst.u.values == 1)
        assert_(nst.a == 1)

    def test_zeros_twopar(self):
        fun1 = Fun(op=lambda x: np.ones_like(x))
        fun2 = Fun(op=lambda x: np.ones_like(x))
        p1 = Parameter(gamma=2)
        p2 = Parameter(beta=3)
        st1  = TwoParameterState(u=fun1, phi=fun2, p1=p1, p2=p2)

        nst = np.zeros_like(st1)

        assert_(st1.u.values == 1)
        assert_(st1.phi.values == 1)
        assert_(st1.b == 2)
        assert_(st1.a == 3)

        assert_(nst.u.values == 0)
        assert_(nst.phi.values == 0)
        assert_(nst.a == 0)
        assert_(nst.b == 0)

    def test_ones_twopar(self):
        fun1 = Fun(op=lambda x: 2 * np.ones_like(x))
        fun2 = Fun(op=lambda x: 2 * np.ones_like(x))
        p1 = Parameter(gamma=2)
        p2 = Parameter(beta=3)
        st1  = TwoParameterState(u=fun1, phi=fun2, p1=p1, p2=p2)

        nst = np.ones_like(st1)

        assert_(st1.u.values == 2)
        assert_(st1.phi.values == 2)
        assert_(st1.b == 2)
        assert_(st1.a == 3)

        assert_(nst.u.values == 1)
        assert_(nst.phi.values == 1)
        assert_(nst.a == 1)
        assert_(nst.b == 1)
