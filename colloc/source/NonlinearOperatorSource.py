#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import itertools
from sympy import Derivative, Integral

from cheb.cbcode import cpcode


class NonlinearFunctionSource:
    """
    Represents the nonlinear integro-differential operator:

        N[u] = f(u, u_x, ..., u_xx, ∫ u)

    """
    def __init__(self, src, *args, **kwargs):
        self.src = src
        self.fun = kwargs.pop('fun', None)
        self.dummy = kwargs.pop('dummy', None)
        self.coeffs = kwargs.pop('coeffs', None)
        self.idx = kwargs.pop('idx', 0)
        self.posInfo = True

        # Collect symbol name
        self.symbol_name = ''

    def __repr__(self):
        return str(self.expr)

    def __str__(self):
        return self.__repr__()

    @property
    def expr(self):
        return self.src

    def emit(self, cg, name, function_names, ftype, domain, *args, **kwargs):
        func = cpcode(self.src, no_evaluation=True)
        self.symbol_name = '{0:s}{1:d}'.format(name, self.idx)

        cg.write('def {0:s}({1}, *args):'.
                 format(self.symbol_name, ', '.join(map(str, function_names))))

        cg.indent()
        cg.write('return asfun({0}, type=\'{1}\', domain=[{2:.16f}, {3:.16f}])'
                 .format(func, ftype, domain[0], domain[1]))

        cg.dedent()
        cg.write('')
