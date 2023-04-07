#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import warnings
import numpy as np
from numbers import Number
from colloc.tools import execute_pycode

from states.tp_state import TwoParameterState


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

    @property
    def pars(self):
        """ Prints the current parameters in the namespace! """
        return self.ns['print_pars']()

    def build(self, src, n_eqn, *args, **kwargs):
        debug = kwargs.get('debug', False)

        # Generate pycode
        pycode = src.emit()

        # Execute code for the rhs
        execute_pycode(pycode, self.ns, debug=debug)

        # Now lookup the required symbols from the namespace
        for func in src.functions:
            try:
                nrhs = self.ns[func.symbol_name]
            except KeyError:
                raise RuntimeError("Could not find \"{0:s}\" in namespace!".
                                   format(func.symbol_name))

            # Append the found executable function!
            self.rhs.append(nrhs)

    def update(self, u, *args, **kwargs):
        n, m = (self.n_disc, u.shape[1])
        self.values = np.zeros((n, m), order='F', dtype=np.complex128 if u.istrig else np.float64)

        # Compute the right hand side values
        for i, rhs in enumerate(self.rhs):
            crhs = rhs(*u, *args, **kwargs)

            # make sure that the function is long enough
            self.values[:, i] = crhs.prolong_coeffs(n)[:, 0]

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
