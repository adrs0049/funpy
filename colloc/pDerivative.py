#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
from ac.support import Namespace

from colloc.chebOpConstraint import ChebOpConstraint
from colloc.tools import execute_pycode, pycode_imports
from colloc.linOp import LinOp
from colloc.rhs import Residual

from states.tp_state import TwoParameterState


class DerivativeFunctional:
    def __init__(self, ns, *args, **kwargs):
        self.ns = ns
        self.n_disc = kwargs.pop('n_disc')
        self.symbol_name = kwargs.pop('symbol_name', 'dp')

        # lookup table for the derivative functions
        self.dps = {}

    def build(self, src, *args, **kwargs):
        execute_pycode(getattr(src, self.symbol_name), self.ns)

        # build the look-up table
        self.collect_symbols_from_ns(src.parameter_names, src.function_names)

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
            return [dp(*u.u, *u.phi) for dp in dps]
        else:
            return [dp(*u) for dp in dps]

    def collect_symbols_from_ns(self, p_names, f_names, *args, **kwargs):
        for p_name in p_names.keys():
            try:
                # create a list for the various partial derivatives
                self.dps[p_name] = [self.ns['{0:s}_{1:s}_{2:s}'\
                                            .format(self.symbol_name, p_name, f_name)] for f_name in f_names]
            except KeyError:
                continue
