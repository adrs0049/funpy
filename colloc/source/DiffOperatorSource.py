#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import itertools
from sympy import Derivative, Function

from cheb.cbcode import cpcode


class DiffOperatorSource:
    """
        Represents the linear operator

                       d^N u
        L[u] = a(x, u) -----
                       dx^N

    """
    def __init__(self, *args, **kwargs):
        self.pos = [None, None]
        self.fun = kwargs.pop('fun', None)
        self.dummy = kwargs.pop('dummy', None)
        self.coeffs = kwargs.pop('coeffs', None)
        self.order = kwargs.pop('order', 0)
        self.posInfo = True

        # Collect symbol name
        self.symbol_name = ''

    def __repr__(self):
        return str(self.expr)

    def __str__(self):
        return self.__repr__()

    @property
    def expr(self):
        if self.order > 0:
            return self.coeffs * Derivative(self.fun, (self.dummy, self.order))
        else:
            return self.coeffs * self.fun

    def emit(self, cg, name, function_names, constant_function_names, ftype, domain, *args, **kwargs):
        ccode = cpcode(self.coeffs,
                       function_names=function_names + constant_function_names, **kwargs)

        # Store symbol name of function
        self.symbol_name = '{0:s}{1:d}{2:d}{3:d}'.\
                format(name, self.pos[0], self.pos[1], self.order)

        # Write to the code gen
        cg.write('')
        cg.write('def {0:s}({1}):'.\
                 format(self.symbol_name, ', '.join(map(str, function_names))))
        cg.indent()

        # Depending on the type of expression we need to write the numpy code differently!
        real = self.coeffs.is_real if self.coeffs.is_real is not None else False
        constant = self.coeffs.is_constant()
        contains_functions = len(self.coeffs.atoms(Function)) != 0

        if self.coeffs.is_zero:
            cg.write('return zeros(1, domain=[{0:.16f}, {1:.16f}], type=\'{2:s}\')'.\
                     format(domain[0], domain[1], ftype))
            self.posInfo = False
        elif constant is not None and (constant or real) and not contains_functions:
            cg.write('return ({0}) * ones(1, domain=[{1:.16f}, {2:.16f}], type=\'{3:s}\')'.\
                     format(ccode, domain[0], domain[1], ftype))
        else:
            cg.write('return asfun({0}, type=\'{1:s}\')'.format(ccode, ftype))

        cg.dedent()
