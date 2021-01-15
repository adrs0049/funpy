#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import sympy as syp
from sympy.printing.pycode import SciPyPrinter, NumPyPrinter


class OdePyPrinter(SciPyPrinter):
    language = "Python with functions"

    _default_settings = dict(
        SciPyPrinter._default_settings,
        no_evaluation=False,
        function_names={})

    def _print_Derivative(self, expr):
        """ Emit python code to compute derivative """
        return r"0.0"

    def _print_Function(self, expr):
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
        elif expr.func.__name__ in self._settings.get('function_names', []):
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
        # elif expr.is_Function and self._settings.get('allow_unknown_functions', False):
        #     return '%s' % self._print(expr.func)
        elif expr.is_Function and self._settings.get('no_evaluation', False):
            return '%s' % (self._print(expr.func))
        else:
            return self._print_not_supported(expr)

    def _print_Integral(self, integral):
        """ Emit python code to compute definite integrals """
        f = integral.function
        return r"%s" % (self._print(f))

def odecode(expr, **settings):
    return OdePyPrinter(settings).doprint(expr)
