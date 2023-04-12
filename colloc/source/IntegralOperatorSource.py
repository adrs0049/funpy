#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
from funpy import cpcode


class IntegralOperatorSource:
    """
        Represents the linear operator

        L[u] = c(x, u) ∫ d(x, u) u(x) dx

    """
    def __init__(self, *args, **kwargs):
        self.fun = kwargs.pop('fun', None)
        self.dummy = kwargs.pop('dummy', None)
        self.coeffs = kwargs.pop('coeffs', None)
        self.integrand = kwargs.pop('integrand', None)
        self.pos = [None, None]
        self.offset = kwargs.pop('offset', 0)
        self.posInfo = True

        self.name_integrand = kwargs.pop('iname', 'ii')
        self.name_coeffs = kwargs.pop('iname', 'ic')

        self.symbol_name_i = ''
        self.symbol_name_c = ''

    def __repr__(self):
        return str(self.expr)

    def __str__(self):
        return self.__repr__()

    @property
    def expr(self):
        return self.coeffs * Integral(self.fun * self.integrand, self.dummy)

    def emit(self, cg, name, function_names, constant_function_names, ftype, domain, *args, **kwargs):
        ccode = cpcode(self.coeffs,
                       function_names=function_names + constant_function_names, **kwargs)

        # Write to the code for the coefficient
        self.symbol_name_c = '{0:s}{1:d}{2:d}{3:s}{4:d}'.\
                format(name, self.pos[0], self.pos[1], self.name_coeffs, self.offset)
        cg.write('')
        cg.write('def {0:s}({1}):'.
                 format(self.symbol_name_c, ', '.join(map(str, function_names))))
        cg.indent()

        if not self.coeffs.is_zero:
            real = self.coeffs.is_real if self.coeffs.is_real is not None else False

            if self.coeffs.is_constant() or real:
                cg.write('return {0} * ones(1, domain=[{1:.16f}, {2:.16f}], type=\'{3:s}\')'.\
                         format(ccode, domain[0], domain[1], ftype))
            else:
                cg.write('return {0}'.format(ccode))

        else:  # nothing -> write zero
            self.posInfo = False
            cg.write('return zeros(1, domain=[{0:.16f}, {1:.16f}], type=\'{2:s}\')'.\
                     format(domain[0], domain[1], ftype))

        cg.dedent()

        # Write to the code for the integrand
        self.symbol_name_i = '{0:s}{1:d}{2:d}{3:s}{4:d}'.\
                format(name, self.pos[0], self.pos[1], self.name_integrand, self.offset)

        # Compile the code
        ccode = cpcode(self.integrand,
                       function_names=function_names + constant_function_names, **kwargs)

        cg.write('')
        cg.write('def {0:s}({1}):'.
                 format(self.symbol_name_i, ', '.join(map(str, function_names))))
        cg.indent()

        if not self.integrand.is_zero:
            real = self.integrand.is_real if self.integrand.is_real is not None else False

            if self.integrand.is_constant() or real:
                cg.write('return {0} * ones(1, domain=[{1:.16f}, {2:.16f}], type=\'{3:s}\')'.\
                         format(ccode, domain[0], domain[1], ftype))
            else:
                cg.write('return {0}'.format(ccode))

        else:  # nothing -> write zero
            self.posInfo = False
            cg.write('return zeros(1, domain=[{0:.16f}, {1:.16f}], type=\'{2:s}\')'.\
                     format(domain[0], domain[1], ftype))

        cg.dedent()
