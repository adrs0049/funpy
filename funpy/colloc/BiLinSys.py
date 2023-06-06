#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np

from .source.support import execute_pycode
from .bilinOp import BiLinOp
from .rhs import Residual
from .pDerivative import DerivativeFunctional
from .LinSysBase import LinSysBase

from funpy.vectorspaces import TwoParameterState


class BiLinSys(LinSysBase):
    """
        Represents bilinear forms resulting from the second derivatives of a
        set of nonlinear equations F(y) = 0.
    """
    def __init__(self, src, diffOrder, *args, **kwargs):
        """ Class that represents a bilinear form -> A[φ, ψ] = b

            where φ is a given fixed function so that this becomes a linear
            system for ψ.
        """
        super().__init__(diffOrder, *args, **kwargs)

        self.linOp     = None   # A
        self.rhs       = None   # b
        self.pDer      = None   # Derivatives of b(u)

        # linear system for potential constraints
        self.c_linOp   = None
        self.c_rhs     = None

        # The Quasi operator
        self.quasiOp = None

        # auto build on construction!
        self.build(src, diffOrder, *args, **kwargs)

    def __str__(self):
        return 'BiLinSys'

    def quasi(self, u, par=False, *args, **kwargs):
        """ Generates the Quasi matrix representing the linear operator
            generated from the bilinear form when φ is fixed.
        """
        assert isinstance(u, TwoParameterState), 'Bilinear form requires a two parameter state! Found %s' % type(u).__name__
        # TODO: assuming here that u and phi contain the same data -> but they
        # would be part of the same state object -> so correct!
        self.update(u)

        # TODO: do this better; it feels like we shouldn't need to do this?
        self.quasiOp = self.linOp.quasi(u.u, u.phi, *args, **kwargs)

        # Add any constraint operators
        if self.c_linOp is not None:
            self.quasiOp += self.c_linOp.quasi(u.u, u.phi, *args, **kwargs)

        return self.quasiOp

    def setDisc(self, n):
        self.n_disc = n
        self.linOp.n_disc = n
        self.rhs.n_disc = n

        if self.c_rhs is not None:
            self.c_rhs.n_disc = n

        if self.c_linOp is not None:
            self.c_linOp.n_disc = n

    def build(self, src, diffOrder=0, matrix_name='fold', dp_name='dxdp', *args, **kwargs):
        # execute the program imports
        execute_pycode(src.common, self.ns)
        assert matrix_name in ['fold', 'bif'], 'Unknown matrix type {}'.format(matrix_name)

        # TODO: replace Create the residual function
        rhs_symbol_name = 'rhs_' + matrix_name
        self.rhs = Residual(self.ns, n_disc=self.n_disc, name=rhs_symbol_name)
        self.rhs.build(getattr(src, rhs_symbol_name), src.symbol_names[rhs_symbol_name],
                       src.n_eqn)

        # Create the second order operator required for fold continuation
        self.linOp = BiLinOp(self.ns, diffOrder)
        self.linOp.build(getattr(src, matrix_name), src.symbol_names[matrix_name])

        # Create the p-derivative function
        if self.par:
            # Compute the derivatives of F'(y) w.r.t. to the two continuation parameters!
            # TODO Also compute the operator w.r.t. to the second continuation parameter!
            self.pDer = DerivativeFunctional(self.ns, n_disc=self.n_disc, symbol_name=dp_name)
            self.pDer.build(src)

        # Build any constraints that may be defined!
        if src.lin_cts is not None:
            cts_symbol_name = 'cts_' + matrix_name

            # Create rhs side for the condition
            self.c_rhs = Residual(self.ns, n_disc=self.n_disc, name=cts_symbol_name)
            self.c_rhs.build(getattr(src, cts_symbol_name), src.symbol_names[cts_symbol_name],
                             src.n_cts)

            # If len(self.conditions) > 0 -> create linear operators for them!
            #self.c_linOp = LinOp(self.ns, diffOrder)
            #self.c_linOp.build(src.lin_cts, src.symbol_names['cts'])

    def getConstraintsRhs(self, u):
        rhs = np.empty((self.numConstraints, 1), dtype=np.float)
        for i, constraint in enumerate(self.constraints):
            rhs[i, 0] = constraint.residual(u)
        return rhs

    def update_partial(self, u, *args, **kwargs):
        """ u: State is expected to have a Namespace providing all required parameters """
        # Update namespace parameters!
        self.setParametersPartial(u)

        self.rhs.update(u)
        if self.c_rhs is not None:
            self.c_rhs.update(u)
            self.rhs.values += self.c_rhs.values

        # update constraint residuals
        try:
            self.cts_res = self.getConstraintsRhs(u.phi)
        except AttributeError:
            self.cts_res = self.getConstraintsRhs(u)

    def update(self, u):
        """ u: State is expected to have a Namespace providing all required parameters """
        self.update_partial(u)
        self.setParameters(u)
