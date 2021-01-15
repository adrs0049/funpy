#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
from numbers import Number
from colloc.tools import execute_pycode


class Residual:
    def __init__(self, ns, *args, **kwargs):
        self.ns = ns
        self.n_disc = kwargs.pop('n_disc')

        # Helpful for debugging to tell various Residuals apart
        self.name = kwargs.pop('name', 'N/A')

        # Storage for prepared function handles
        self.rhs = kwargs.pop('rhs', [])

        # store the current rhs
        self.values = kwargs.pop('values', None)

    def build(self, src, symbol_name, *args, **kwargs):
        debug = kwargs.get('debug', False)

        # Execute code for the rhs
        execute_pycode(src, self.ns, debug=debug)
        self.collect_symbols_from_ns(lambda idx: '{0}{1}'.format(symbol_name, idx), self.rhs)

    def update(self, u):
        n, m = (self.n_disc, u.shape[1])
        self.values = np.zeros((n, m), order='F', dtype=np.complex128 if u.istrig else np.float64)

        # Compute the right hand side values
        for i, rhs in enumerate(self.rhs):
            crhs = rhs(*u)

            # make sure that the function is long enough
            self.values[:, i] = crhs.prolong_coeffs(n)[:, 0]

    def collect_symbols_from_ns(self, symbol_name, dest):
        idx = 0
        while True:
            try:
                func = self.ns[symbol_name(idx)]
                dest.append(func)
                idx += 1
            except KeyError:
                break

    def __pos__(self):
        return self

    def __neg__(self):
        return Residual(self.ns, n_disc=self.n_disc, values=-self.values)

    def __add__(self, other):
        assert type(self) == type(other), ''
        if self.values.shape != other.values.shape:
            raise ValueError("")

        new_values = np.copy(self.values) + other.values
        return Residual(self.ns, n_disc=self.n_disc, values=new_values)

    def __sub__(self, other):
        return self.__add__(-other)

    def __mul__(self, other):
        if not isinstance(other, Number):
            raise ValueError("Residual.__mul__ other must be a real!")

        return Residual(self.ns, n_disc=self.n_disc, values=other * self.values)

    def __rmul__(self, other):
        return other * self

    def __iadd__(self, other):
        raise RuntimeError("+= is not allowed for LinBlock!")
