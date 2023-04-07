#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
from cheb.cbcode import cpcode


class BoundaryConditionSource:
    """
        Represents a pair of boundary conditions.

    """
    def __init__(self, *args, **kwargs):
        self.fun = kwargs.pop('fun', None)
        self.dummy = kwargs.pop('dummy', None)
        self.coeffs = kwargs.pop('coeffs', None)
        self.location = kwargs.pop('location', 0.0)

        # Collect symbol name
        self.symbol_name = ''

    def __repr__(self):
        return str(self.expr)

    def __str__(self):
        return self.__repr__()

    @property
    def expr(self):
        return self.expr

    def emit(self, cg, name, function_names, constant_function_names,
             ftype, domain, *args, **kwargs):

        ccode = cpcode(self.coeffs,
                       #function_names=function_names + constant_function_names,
                       **kwargs)

        # Store symbol name of function
        self.symbol_name = '{0:s}'.format(name)

        # Write to the code gen
        #
        # TODO: A current limitation of the chebcolloc2 code is that chebcollocs
        # do not form a field. Thus we need to make sure that the evaluation operators
        # are distributed among equation terms.
        #
        cg.write('')
        cg.write('def {0:s}({1}):'.\
                 format(self.symbol_name, ', '.join(map(str, function_names))))
        cg.indent()

        # HACK: make sure we replace the default spatial variable with the location at
        # which we want to evaluate the functions.
        ccode = ccode.replace('x', '{0:.16f}'.format(self.location))
        cg.write('return {0}'.format(ccode))
        cg.dedent()
