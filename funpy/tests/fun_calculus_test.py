#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
from numpy.testing import assert_, assert_raises, assert_almost_equal
from fun import Fun, norm, norm2, h1norm, wkpnorm


class TestFunCalculus:
    def test_diff(self):
        f  = Fun(op=lambda x: np.sin(2 * np.pi * x), type='cheb')
        co = np.copy(f.coeffs)
        df = Fun(op=lambda x: 2*np.pi * np.cos(2 * np.pi * x), type='cheb')

        # test domain
        xs = np.linspace(-1, 1, 100)

        # compute derivative
        ddf = np.diff(f)

        # Make sure f is unchanged
        assert_almost_equal(co, f.coeffs)

        # Just check the values it returns
        assert_almost_equal(ddf(xs), df(xs))

    def test_diff_domain(self):
        f  = Fun(op=lambda x: np.sin(2 * np.pi * x), type='cheb')
        co = np.copy(f.coeffs)
        df = Fun(op=lambda x: 2*np.pi * np.cos(2 * np.pi * x), type='cheb', domain=[0, 1])

        # test domain
        xs = np.linspace(0, 1, 100)

        # compute derivative
        ddf = np.diff(f)

        # Make sure f is unchanged
        assert_almost_equal(co, f.coeffs)

        # Just check the values it returns
        assert_almost_equal(ddf(xs), df(xs))

    def test_cumsum(self):
        f  = Fun(op=lambda x: np.sin(2 * np.pi * x), type='cheb')
        df = Fun(op=lambda x: (1 - np.cos(2 * np.pi * x)) / (2 * np.pi), type='cheb')

        # test domain
        xs = np.linspace(-1, 1, 100)

        # compute derivative
        cf = np.cumsum(f)

        # Just check the values it returns
        assert_almost_equal(cf(xs), df(xs))

    def test_sum(self):
        f = Fun(op=lambda x: np.sin(2 * np.pi * x), type='cheb')
        g = Fun(op=lambda x: np.sin(10 * np.pi * x), type='cheb')
        h = Fun(op=lambda x: np.sin(20 * np.pi * x), type='cheb')
        assert_almost_equal(np.sum(f), 0)
        assert_almost_equal(np.sum(g), 0)
        assert_almost_equal(np.sum(h), 0)

    def test_sum2(self):
        f  = Fun(op=lambda x: x**2, type='cheb')
        assert_almost_equal(np.sum(f), 2./3)

    def test_norm(self):
        f = Fun(op=lambda x: np.sin(2 * np.pi * x), type='cheb')
        assert_almost_equal(norm(f), 1.0)
        # TODO: figure out why the accuracy of the L1-norm is worse
        assert_almost_equal(norm(f, p=2), 1.0)
        assert_almost_equal(norm(f, p=4), (3./4)**0.25)
        assert_almost_equal(norm(f, p=6), (5./8)**(1./6))

    def test_norm_odd(self):
        # TODO: FIX! These currently fail since they are no longer smooth
        f = Fun(op=lambda x: np.sin(2 * np.pi * x), type='cheb')
        assert_almost_equal(norm(f, p=1), 4. / np.pi, decimal=4)
        assert_almost_equal(norm(f, p=3), (8. / (3. * np.pi))**(1./3))
        assert_almost_equal(norm(f, p=5), (32 / (15. * np.pi))**(1./5))

    def test_norm2(self):
        f = Fun(op=lambda x: np.sin(2 * np.pi * x), type='cheb')
        assert_almost_equal(norm2(f), 1.0)
        # TODO: figure out why the accuracy of the L1-norm is worse
        assert_almost_equal(norm2(f, p=2), 1.0)
        assert_almost_equal(norm2(f, p=4), 3./4)
        assert_almost_equal(norm2(f, p=6), 5./8)

    def test_norm2_odd(self):
        f = Fun(op=lambda x: np.sin(2 * np.pi * x), type='cheb')
        assert_almost_equal(norm2(f, p=1), 4. / np.pi, decimal=4)
        assert_almost_equal(norm2(f, p=3), 8. / (3. * np.pi))
        assert_almost_equal(norm2(f, p=5), 32 / (15. * np.pi))

    def test_h1norm(self):
        f = Fun(op=lambda x: np.sin(2 * np.pi * x), type='cheb')
        assert_almost_equal(h1norm(f), (1 + 4. * np.pi**2)**0.5)
        assert_almost_equal(h1norm(f, p=2), (1 + 4. * np.pi**2)**0.5)
        assert_almost_equal(wkpnorm(f, p=2, k=0), 1)
        assert_almost_equal(wkpnorm(f, p=2, k=1), (1 + 4. * np.pi**2)**0.5)
        assert_almost_equal(wkpnorm(f, p=2, k=2), (1 + 4. * np.pi**2 + 16 * np.pi**4)**0.5)
        assert_almost_equal(wkpnorm(f, p=2, k=3), (1 + 4. * np.pi**2 + 16 * np.pi**4 + 64*np.pi**6)**0.5)
