#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
from vectorspaces import Namespace

from .source.support import execute_pycode
from .rhs import Residual


class Function:
    """
        TODO: write me!
    """
    def __init__(self, src, *args, **kwargs):
        self.ns = Namespace()
        self.n_disc = kwargs.get('n_disc', 15)

        # linear system for potential constraints
        self.rhs = None

        # auto build on construction!
        self.build(src, *args, **kwargs)

    @property
    def pars(self):
        """ Prints the current parameters in the namespace! """
        return self.ns['print_pars']()

    def __call__(self, *args, **kwargs):
        self.rhs.update(*args, **kwargs)
        return self.rhs.values.flatten(order='F')

    def update_partial(self, u, *args, **kwargs):
        """ u: State is expected to have a Namespace providing all required parameters """
        # Update namespace parameters!
        self.setParametersPartial(u)

    def update(self, u):
        """ u: State is expected to have a Namespace providing all required parameters """
        self.update_partial(u)
        self.setParameters(u)

    def build(self, src, matrix_name='fold', dp_name='dxdp', *args, **kwargs):
        # execute the program imports
        execute_pycode(src.common, self.ns)

        self.rhs = Residual(self.ns, n_disc=self.n_disc, name='rhs_curv')
        self.rhs.build(getattr(src, 'rhs_curv'), src.symbol_names['rhs_curv'], src.n_cts)

    def setParametersPartial(self, pars):
        # update the namespace
        try:
            for p_name, p_value in pars.update_items():
                self.ns[p_name] = p_value
        except AttributeError:
            pass

    def setParameters(self, pars):
        # update the namespace
        try:
            for p_name, p_value in pars.items():
                self.ns[p_name] = p_value
        except AttributeError:
            pass
