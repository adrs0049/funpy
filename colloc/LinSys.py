#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np

from .chebOpConstraint import ChebOpConstraint
from .chebOpConstraintCompiled import ChebOpConstraintCompiled
from .source.support import execute_pycode
from .LinSysBase import LinSysBase
from .linOp import LinOp
# from .bilinOp import BiLinOp
from .rhs import Residual
from .linearFunctional import linearFunctional
from .bilinearForm import bilinearForm
from .pDerivative import DerivativeFunctional


class LinSys(LinSysBase):
    """
    Represents various linearizations of a nonlinear set of equations F(y) = 0.
    """
    def __init__(self, src, diffOrder: int, *args, **kwargs):
        """ Class that represents a linear system -> Ax = b """
        super().__init__(diffOrder, *args, **kwargs)

        self.linOp = None      # A
        self.rhs   = None      # b
        self.df    = None      # First derivative
        self.ddx   = None      # Second derivative bilinear forms
        self.pDer  = None      # Derivatives of b(u)
        self.dxdp  = None
        self.linOp_bif = None  # Operator required for Moore form

        # linear system for potential constraints
        self.c_linOp = None
        self.c_rhs   = None

        # The Quasi operator
        self.quasiOp = None

        # auto build on construction!
        self.build(src, diffOrder, **kwargs)

    def __str__(self):
        return 'LinSys'

    def quasi(self, u, full=True, par=False, *args, **kwargs):
        """ Generates the Quasi matrix representing this linear operator """
        self.update(u)

        # TODO: FIXME again why the sub-selection of the object?
        try:
            self.quasiOp = self.linOp.quasi(u.u, *args, **kwargs)
            if self.c_linOp is not None:
                self.quasiOp += self.c_linOp.quasi(u.u, *args, **kwargs)

        except AttributeError:
            self.quasiOp = self.linOp.quasi(u, *args, **kwargs)

            # Add any constraint operators
            if self.c_linOp is not None:
                self.quasiOp += self.c_linOp.quasi(u, *args, **kwargs)

        return self.quasiOp

    def setDisc(self, n):
        self.n_disc = n
        self.linOp.n_disc = n
        self.rhs.n_disc = n
        if self.df is not None: self.df.n_disc = n
        if self.ddx is not None: self.ddx.n_disc = n
        if self.par: self.pDer.n_disc = n

        if self.c_linOp is not None:
            self.c_linOp.n_disc = n
            self.c_rhs.n_disc = n

    def build(self, src, diffOrder=0, *args, **kwargs):
        debug = kwargs.get('debug', False)

        # execute the program imports
        execute_pycode(src.common, self.ns, debug=debug)

        # Create the linear operator
        self.linOp = LinOp(self.ns, diffOrder)
        self.linOp.build(src.lin, src.symbol_names['eqn'], src.symbol_names['adh'], **kwargs)

        # Create the residual function
        self.rhs = Residual(self.ns, n_disc=self.n_disc, name='rhs')
        self.rhs.build(src.eqn, src.n_eqn, **kwargs)

        # Create the second derivative vector bilinear form
        if self.par and (src.deqn is not None):
            # The action of the Frechet derivative of the operator
            self.df = linearFunctional(self.ns, n_disc=self.n_disc, name='df')
            self.df.build(src.deqn, src.n_eqn, **kwargs)

            # The action of the bilinear form of the second order
            # Frechet derivative of the operator.
            self.ddx = bilinearForm(self.ns, n_disc=self.n_disc, name='ddx')
            self.ddx.build(src.ddeqn, src.n_eqn, **kwargs)

            # The derivatives of the linearization
            self.dxdp  = linearFunctional(self.ns, **kwargs)
            self.dxdp.build(src.dxdp, src.n_eqn, **kwargs)

            # The action of the bilinear form of the adjoint second order
            # Frechet derivative of the operator.
            self.ddx_adj = bilinearForm(self.ns, n_disc=self.n_disc, name='ddx_adj')
            self.ddx_adj.build(src.dd_adjeqn, src.n_eqn, **kwargs)

        if self.par:
            # Create the p-derivative function
            # The derivative of the operator w.r.t. the main continuation parameter
            self.pDer = DerivativeFunctional(self.ns, **kwargs)
            self.pDer.build(src, **kwargs)

        # Build any constraints that may be defined!
        if src.lin_cts is not None:
            # If len(self.conditions) > 0 -> create linear operators for them!
            self.c_linOp = LinOp(self.ns, diffOrder)
            self.c_linOp.build(src.lin_cts, src.symbol_names['cts'], **kwargs)

            # Create rhs side for the condition
            self.c_rhs = Residual(self.ns, n_disc=self.n_disc, name='crhs')
            self.c_rhs.build(src.cts, src.n_cts, **kwargs)
            def pp(u):
                return u
            self.c_rhs.proj = lambda u: pp(u)

        # Compile constraints
        # 1. Execute the constraint code in the namespace
        pycode = src.bcs.emit()
        execute_pycode(pycode, self.ns, debug=debug)

        for bc_op in src.bcs:
            try:
                constraint = ChebOpConstraint(op=self.ns[bc_op.symbol_name],
                                              domain=src.domain)

                self.constraints.append(constraint)
            except KeyError:
                raise RuntimeError('Could not find {0:s} in the namespace!'.format(bc_op.symbol_name))

        # Compile the boundary cache!
        bc_cache = ChebOpConstraintCompiled(self, n=1 + 2**12, m=src.n_eqn, domain=src.domain)
        blocks = bc_cache.compile()

        # Assign the compiled BCs
        for i, constraint in enumerate(self.constraints):
            constraint.compiled = blocks[i]

    def getConstraintsRhs(self, u):
        rhs = np.empty((self.numConstraints, 1), dtype=np.float)
        for i, constraint in enumerate(self.constraints):
            rhs[i, 0] = constraint.residual(u)
        return rhs

    def update_partial(self, u):
        """ u: State is expected to have a Namespace providing all required parameters """
        if u is None: return

        # Update namespace parameters!
        self.setParametersPartial(u)

        # Update the residual
        self.rhs.update(u)
        if self.c_rhs is not None:
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
