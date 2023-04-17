#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
from numpy.testing import assert_, assert_raises

import funpy as fp
from funpy import Fun


class TestFunAddition:
    def test_addition(self):
        fun1 = Fun(op=lambda x: np.ones_like(x), type='cheb')
        fun2 = Fun(op=lambda x: np.ones_like(x), type='cheb')
        fun3 = fun1 + fun2

        assert_(fun3.values == 2)
        # make sure that original functions were unchanged!
        assert_(fun1.values == 1)
        assert_(fun2.values == 1)

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


if __name__ == '__main__':
    test = TestFunAddition()
    test.test_addition_scalar()
