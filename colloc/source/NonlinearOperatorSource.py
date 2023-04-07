#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import itertools
from sympy import Derivative, Integral

from ac.gen import CodeGeneratorBackend

from cheb.cbcode import cpcode


class NonlinearFunctionSource:
    """
    Represents the nonlinear integro-differential operator:

        N[u] = f(u, u_x, ..., u_xx, ∫ u)

    """
    def __init__(self, src, *args, **kwargs):
        self.src = src
        self.name = kwargs.pop('name', '')
        self.function_names = kwargs.pop('functions', None)
        self.fun = kwargs.pop('fun', None)
        self.dummy = kwargs.pop('dummy', None)
        self.coeffs = kwargs.pop('coeffs', None)
        self.idx = kwargs.pop('idx', 0)
        self.posInfo = True

        # Some stuff to set
        self.ftype  = kwargs.pop('ftype', None)
        self.domain = kwargs.pop('domain', None)

        # Collect symbol name
        self.symbol_name = ''

    def __repr__(self):
        return str(self.expr)

    def __str__(self):
        return self.__repr__()

    @property
    def expr(self):
        return self.src

    # TODO: Improve this hack eventually!
    def emit_detail(self, cg, name, function_names, ftype, domain, *args, **kwargs):
        func = cpcode(self.src, no_evaluation=True)
        self.symbol_name = '{0:s}{1:d}'.format(name, self.idx)

        cg.write('def {0:s}({1}, *args):'.
                 format(self.symbol_name, ', '.join(map(str, function_names))))

        cg.indent()
        cg.write('return asfun({0}, type=\'{1}\', domain=[{2:.16f}, {3:.16f}])'
                 .format(func, ftype, domain[0], domain[1]))

        cg.dedent()
        cg.write('')

    def emit(self, *args, **kwargs):
        cg = CodeGeneratorBackend()
        cg.begin(tab=4*" ")
        cg.write(35 * '#')
        cg.write('# Function block {0:s}.'.format(self.name))
        cg.write(35 * '#')

        self.emit_detail(cg, self.name, self.function_names,
                         self.ftype, self.domain, *args, **kwargs)

        return cg.end()
