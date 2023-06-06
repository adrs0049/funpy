#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np

from funpy.vectorspaces import Namespace, TwoParameterState

from .chebOpConstraint import ChebOpConstraint
from .source.support import execute_pycode
from .linOp import LinOp
from .rhs import Residual


class DerivativeFunctional:
    def __init__(self, ns, *args, **kwargs):
        self.ns = ns
        self.n_disc = kwargs.pop('n_disc')
        self.symbol_name = kwargs.pop('symbol_name', 'dp')

        # lookup table for the derivative functions
        self.dps = {}

    def build(self, src, *args, **kwargs):
        for p_name, vfunc in src.dp.items():

            # Generate the python code
            pycode = vfunc.emit()

            # Execute the code
            execute_pycode(pycode, self.ns)

            # Get the symbols from the namespace
            self.dps[p_name] = []
            for func in vfunc.functions:
                try:
                    symbol = self.ns[func.symbol_name]
                    self.dps[p_name].append(symbol)

                except KeyError:
                    raise RuntimeError("Could not find symbol \"{}\"!".format(func.symbol_name))

    def __call__(self, u, pname=None):
        """ Computes the derivative of the nonlinear operator with respect to a
            which must be one of the known parameters.

            Since the result of this will be added to the result of D_u F(a, u) so it must be in
            the same basis as that vector!
        """
        # Compute the coefficients of D_a and apply change of basis matrix + projection + re-sort!
        if pname is None: pname = u.cpar

        try:
            dps = self.dps[pname]
        except KeyError:
            raise KeyError("Unknown parameter {0:s}!".format(pname))

        if isinstance(u, TwoParameterState):
            return np.hstack([dp(*u.u, *u.phi) for dp in dps])
        else:
            return np.hstack([dp(*u) for dp in dps])
