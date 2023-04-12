#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np

from .source.support import execute_pycode


class bilinearForm:
    def __init__(self, ns, *args, **kwargs):
        self.ns = ns
        self.n_disc = kwargs.pop('n_disc')

        # Helpful for debugging to tell various Residuals apart
        self.name = kwargs.pop('name', 'N/A')

        # Storage for prepared function handles
        self.functions = kwargs.pop('function', [])

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
            self.functions.append(nrhs)

    def __call__(self, u, up, upp, *args, **kwargs):
        return np.hstack([func(*u, *up, *upp) for func in self.functions])
