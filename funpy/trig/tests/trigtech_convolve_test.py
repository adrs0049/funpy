#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
from numpy.testing import assert_, assert_raises, assert_almost_equal

import funpy as fp
from funpy import Fun
from funpy.trig import trigtech


class TestTrigtechConvolution:
    def test_shift(self):
        f = Fun(op=lambda x: np.sin(2 * np.pi * x), type='trig')

        # test domain
        xs = np.linspace(-1, 1, 100)

        # test #1
        avalues = np.arange(-1, 1.01, 0.1)
        for a in avalues:
            g = np.roll(f, a)
            g_expected = lambda x: np.sin(2. * np.pi * (x - a))

            # Just check the values it returns
            assert_almost_equal(g(xs), g_expected(xs))

    def test_convolve(self):
        f = Fun(op=lambda x: np.sin(x), domain=[-np.pi, np.pi], type='trig')
        g = Fun(op=lambda x: np.cos(x), domain=[-np.pi, np.pi], type='trig')

        # compute convolution
        h = np.convolve(f, g)

        # test domain
        xs = np.linspace(-np.pi, np.pi, 1000)

        # Just check the values it returns
        he = lambda x: np.pi * np.sin(x)
        assert_almost_equal(h(xs), he(xs))

    # def test_nonlocal_op(self):
    #     """
    #         EXPECTED TO FAIL RIGHT NOW!
    #     """
    #     # Tests the non-local operator appearing in non-local adhesion models
    #     R = 1
    #     L = 3
    #     n = 2

    #     # The coefficient that appears
    #     M = lambda n: L * np.sin(np.pi * n / L)**2 / (2. * np.pi * n)

    #     domain = [-0.5*L, 0.5*L]
    #     omega  = lambda x: np.piecewise(x, [x < -R, (x>=-R)&(x<=0), (x>0)&(x<=R), x>R],
    #                                     [lambda x: np.zeros_like(x), lambda x: -0.5 * np.ones_like(x), lambda x: 0.5 * np.ones_like(x), lambda x: np.zeros_like(x)])

    #     # need to flip this since the convolution has a particular sign input for the kernel
    #     kernel = Fun(op=lambda x: omega(-x), type='trig', domain=domain)

    #     xs     = np.linspace(domain[0], domain[1], 1000)

    #     test_function1 = Fun(op=lambda x: np.sin(2.*np.pi * n * x/ L), domain=domain, type='trig')
    #     test_function2 = Fun(op=lambda x: np.cos(2.*np.pi * n * x/ L), domain=domain, type='trig')

    #     expected1 = Fun(op=lambda x:   2. * M(n) * np.cos(2.*np.pi * n * x/ L), domain=domain, type='trig')
    #     expected2 = Fun(op=lambda x: - 2. * M(n) * np.sin(2.*np.pi * n * x/ L), domain=domain, type='trig')

    #     # compute the two outputs
    #     num1 = np.convolve(test_function1, kernel)
    #     num2 = np.convolve(test_function2, kernel)

    #     # Just check the values it returns
    #     assert_almost_equal(num1(xs), expected1(xs))
    #     assert_almost_equal(num2(xs), expected2(xs))
