#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import itertools
from sympy import Derivative, Integral

from ac.gen import CodeGeneratorBackend

from cheb.cbcode import cpcode
from colloc.source.NonlinearOperatorSource import NonlinearFunctionSource


class NonlinearVectorFunctionSource:
    """
    Represents a vector nonlinear integro-differential operator:

        N[u] = f(u, u_x, ..., u_xx, ∫ u)

    """
    def __init__(self, src, ftype, domain, *args, **kwargs):
        self.function_names = kwargs.pop('functions', None)
        # self.dummy = kwargs.pop('dummy', None)

        # Store the functions building the vector function.
        self.functions = []

        # get name
        self.name = kwargs.pop('name', '')

        # Some stuff to set
        self.ftype = ftype
        self.domain = domain

        for i in range(src.shape[0]):
            new_function = NonlinearFunctionSource(src[i, 0], idx=i)
            self.functions.append(new_function)

    def __repr__(self):
        rstr = type(self).__name__ + ' \"{0:s}\"'.format(self.name) + ':\n'
        for func in self.functions:
            rstr += repr(func)
        return rstr

    def __str__(self):
        return self.__repr__()

    def emit(self, *args, **kwargs):
        cg = CodeGeneratorBackend()
        cg.begin(tab=4*" ")
        cg.write(35 * '#')
        cg.write('# Function block {0:s}.'.format(self.name))
        cg.write(35 * '#')

        for func in self.functions:
            func.emit(cg, self.name, self.function_names, self.ftype,
                      self.domain, *args, **kwargs)

        return cg.end()
