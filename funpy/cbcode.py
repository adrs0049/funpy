#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import sympy as syp
from sympy import Function

try:
    # Newer versions of sympy seem to have moved this
    from sympy.printing.pycode import SciPyPrinter, NumPyPrinter
except ImportError:
    from sympy.printing.numpy import SciPyPrinter, NumPyPrinter


class FunPyPrinter(SciPyPrinter):
    language = "Python with funpy"

    _default_settings = dict(
        SciPyPrinter._default_settings,
        no_evaluation=False,
        function_names={})

    def _print_Subs(self, expr):
        expr, new, old = expr.args
        self._settings['no_evaluation'] = True
        return r'{0:s}({1:s})'.format(self._print(expr), self._print(old[0]))

    def _print_Derivative(self, expr, no_evaluation=False):
        """ Emit python code to compute derivative """
        dim = 0
        arg = None
        for x, num in reversed(expr.variable_count):
            arg = x
            dim += num

        # print('func_name:', expr.expr.func.__name__, ' args:', expr.expr.args)
        # print('function_names:', self._settings.get('function_names'))
        # since any sub expression is inside the diff it should not be evaluated
        # expr.expr.args = tuple([])

        if self._settings.get('no_evaluation', False) or no_evaluation:
            return r"np.diff(%s, %d)" % (self._print(expr.expr), dim)
        else:
            return r"np.diff(%s, %d)(%s)" % (self._print(expr.expr, overwrite_eval=True), dim, arg)

    def _print_Function(self, expr, overwrite_eval=False):
        if expr.func.__name__ in self.known_functions:
            cond_func = self.known_functions[expr.func.__name__]
            func = None
            if isinstance(cond_func, str):
                func = cond_func
            else:
                for cond, func in cond_func:
                    if cond(*expr.args):
                        break
            if func is not None:
                try:
                    return func(*[self.parenthesize(item, 0) for item in expr.args])
                except TypeError:
                    return "%s(%s)" % (func, self.stringify(expr.args, ", "))
        elif (expr.func.__name__ in self._settings.get('function_names', [])) or overwrite_eval:
            return '%s' % self._print(expr.func)
        elif hasattr(expr, '_imp_') and isinstance(expr._imp_, Lambda):
            # inlined function
            return self._print(expr._imp_(*expr.args))
        elif (expr.func.__name__ in self._rewriteable_functions and
              self._rewriteable_functions[expr.func.__name__] in self.known_functions):
            # Simple rewrite to supported function possible
            return self._print(expr.rewrite(self._rewriteable_functions[expr.func.__name__]))
        elif expr.is_Function and self._settings.get('allow_unknown_functions', False):
            return '%s(%s)' % (self._print(expr.func), ', '.join(map(self._print, expr.args)))
        elif expr.is_Function and self._settings.get('no_evaluation', False):
            return '%s' % (self._print(expr.func))
        else:
            return self._print_not_supported(expr)

    def _print_Integral(self, integral):
        """ Emit python code to compute definite integrals

            This code emits different constructs for convolutions and linear functionals.
        """
        # We make the following assumption
        if all(len(lim) == 1 for lim in integral.limits):
            # Treat it as an integral constraint
            return r"np.sum(%s)" % (self._print(integral.function))

        else:  # We have integration limits -> map to a convolution
            integrands = list(integral.atoms(Function))
            n = len(integrands)

            # For a convolution we need two functions
            if n == 2:
                kernel = None
                argument = None
                for integrand in integrands:
                    if len(integrand.free_symbols) == 1:
                        argument = integrand
                    elif len(integrand.free_symbols) == 2:
                        kernel = integrand
                    else:
                        return NotImplemented

                # TODO: somehow solve this better in the future!
                return r"adhesion(%s, %s)" % (self._print(argument), self._print(kernel))

            # cumsum??
            elif n == 1:
                return r"np.cumsum(%s)" % (self._print(integral.function))
            else:
                return NotImplemented

def npcode(expr, **settings):
    return NumPyPrinter(settings).doprint(expr)

def spcode(expr, **settings):
    return SciPyPrinter(settings).doprint(expr)

def cpcode(expr, **settings):
    return FunPyPrinter(settings).doprint(expr)
