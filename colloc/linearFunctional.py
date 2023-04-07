#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import warnings
import numpy as np
from numbers import Number
from colloc.tools import execute_pycode

from states.tp_state import TwoParameterState


class linearFunctional:
    def __init__(self, ns, *args, **kwargs):
        self.ns = ns
        self.n_disc = kwargs.pop('n_disc')

        # Helpful for debugging to tell various Residuals apart
        self.name = kwargs.pop('name', 'N/A')

        # Lookup table for functions
        self.functions = {}

        # call function
        self._call = None

    @property
    def pars(self):
        """ Prints the current parameters in the namespace! """
        return self.ns['print_pars']()

    def build(self, src, n_eqn, *args, **kwargs):
        debug = kwargs.get('debug', False)

        print('src = ', type(src))
        print(src)
        if isinstance(src, dict):
            self.functions = {}
            for p_name, vfunc in src.items():
                # Generate pycode
                pycode = vfunc.emit()

                # Execute code for the rhs
                execute_pycode(pycode, self.ns, debug=debug)

                # Now lookup the required symbols from the namespace
                self.functions[p_name] = []
                for func in vfunc.functions:
                    try:
                        symbol = self.ns[func.symbol_name]
                        self.functions[p_name].append(symbol)
                    except KeyError:
                        raise RuntimeError("Could not find \"{0:s}\" in namespace!".
                                           format(func.symbol_name))

            # Set the call to dictionary
            self._call = self.call_dict

        else:  # TODO check that it's simple?
            # Generate pycode
            pycode = src.emit()

            # Execute the code for the rhs
            execute_pycode(pycode, self.ns, debug=debug)

            # Now lookup the required symbols from the namespace
            self.functions = []
            for func in src.functions:
                try:
                    nrhs = self.ns[func.symbol_name]
                except KeyError:
                    raise RuntimeError("Could not find \"{0:s}\" in namespace!".
                                       format(func.symbol_name))

                # Append the found executable function!
                self.functions.append(nrhs)

            # Use the list call method
            self._call = self.call_list

    def call_dict(self, u, up, pname=None, *args, **kwargs):
        if pname is None: pname = u.cpar

        try:
            funcs = self.functions[pname]
        except KeyError:
            raise KeyError("Unknown parameter {0:s}!".format(pname))

        return np.hstack([func(*u, *up) for func in funcs])

    def call_list(self, u, up, *args, **kwargs):
        return np.hstack([func(*u, *up) for func in self.functions])

    def __call__(self, u, up, pname=None, *args, **kwargs):
        return self._call(u, up, *args, **kwargs)
