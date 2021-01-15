#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
from ac.support import Namespace
from fun import Fun

from colloc.chebOpConstraint import ChebOpConstraint
from colloc.tools import execute_pycode
from colloc.LinSysBase import LinSysBase
from colloc.linOp import LinOp
from colloc.bilinOp import BiLinOp
from colloc.rhs import Residual
from colloc.pDerivative import DerivativeFunctional


class LinSys(LinSysBase):
    """
    Represents various linearizations of a nonlinear set of equations F(y) = 0.
    """
    def __init__(self, src, diffOrder: int, *args, **kwargs):
        """ Class that represents a linear system -> Ax = b """
        super().__init__(diffOrder, *args, **kwargs)

        self.linOp = None   # A
        self.rhs   = None   # b
        self.pDer  = None   # Derivatives of b(u)
        self.linOp_bif = None # Operator required for Moore form

        # linear system for potential constraints
        self.c_linOp = None
        self.c_rhs   = None

        # The Quasi operator
        self.quasiOp = None

        # auto build on construction!
        self.build(src, diffOrder)

    def __str__(self):
        return 'LinSys'

    def quasi(self, u, par=False):
        """ Generates the Quasi matrix representing this linear operator """
        self.update(u)

        # TODO: FIXME again why the sub-selection of the object?
        try:
            self.quasiOp = self.linOp.quasi(u.u)
        except AttributeError:
            self.quasiOp = self.linOp.quasi(u)

        # Add any constraint operators
        if self.c_linOp is not None:
            self.quasiOp += self.c_linOp.quasi(u.u)

        return self.quasiOp

    def setDisc(self, n):
        self.n_disc = n
        self.linOp.n_disc = n
        self.rhs.n_disc = n
        if self.par: self.pDer.n_disc = n

        if self.c_linOp is not None:
            self.c_linOp.n_disc = n
            self.c_rhs.n_disc = n

    def build(self, src, diffOrder=0, *args, **kwargs):
        # execute the program imports
        execute_pycode(src.common, self.ns)

        # Create the linear operator
        self.linOp = LinOp(self.ns, diffOrder)
        self.linOp.build(src.lin, src.symbol_names['eqn'], src.symbol_names['adh'])

        # Create the residual function
        self.rhs = Residual(self.ns, n_disc=self.n_disc, name='rhs')
        self.rhs.build(src.eqn, src.symbol_names['eqn'])

        # Create the p-derivative function
        if self.par:
            # The derivative of the operator w.r.t. the main continuation parameter
            self.pDer = DerivativeFunctional(self.ns, n_disc=self.n_disc)
            self.pDer.build(src)

        # Build any constraints that may be defined!
        if src.lin_cts is not None:
            # If len(self.conditions) > 0 -> create linear operators for them!
            self.c_linOp = LinOp(self.ns, diffOrder)
            self.c_linOp.build(src.lin_cts, src.symbol_names['cts'])

            # Create rhs side for the condition
            self.c_rhs = Residual(self.ns, n_disc=self.n_disc, name='crhs')
            self.c_rhs.build(src.cts, src.symbol_names['cts'])
            def pp(u):
                return u
            self.c_rhs.proj = lambda u: pp(u)

    def getConstraintsRhs(self, u):
        rhs = np.empty((self.numConstraints, 1), dtype=np.float)
        for i, constraint in enumerate(self.constraints):
            rhs[i, 0] = constraint.residual(u)
        return rhs

    def update_partial(self, u):
        """ u: State is expected to have a Namespace providing all required parameters """
        # Update namespace parameters!
        self.setParametersPartial(u)

        # Update the residual
        self.rhs.update(u)
        if self.c_linOp is not None:
            self.c_rhs.update(u)
            self.rhs.values += self.c_rhs.values

        # update constraint residuals
        try:
            self.cts_res = self.getConstraintsRhs(u.u)
        except AttributeError:
            self.cts_res = self.getConstraintsRhs(u)

    def update(self, u):
        """ u: State is expected to have a Namespace providing all required parameters """
        self.setParameters(u)
        self.update_partial(u)
