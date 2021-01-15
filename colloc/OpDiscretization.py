#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
import scipy as sp
import scipy.sparse as sps
import scipy.linalg as LA
import scipy.sparse.linalg as LAS
from scipy.sparse import eye, bmat

from sparse.csr import flip_rows

from funpy.fun import Fun
from funpy.mapping import Mapping

from funpy.cheb.chebpy import chebtec
from funpy.colloc.ultraS.matrices import blockmat
from funpy.colloc.projection import RectangularProjection


class OpDiscretization(object):
    """ Converts an infinite dimensional operator to discrete form """
    def __init__(self, source=None, *args, **kwargs):
        self.dimension = np.atleast_1d(kwargs.get('dimension', 1))
        self.domain = np.asarray(kwargs.get('domain', [-1, 1]))
        self.numIntervals = self.domain.size - 1

        # the linear operator that we are discretizing
        self.source = source

        # create a domain mapping
        self.mapping = Mapping(ends=self.domain)

        # TEMP -> these should be a part of the source!
        self.constraints = self.source.constraints if self.source is not None else []
        self.continuity  = kwargs.pop('continuity', [])

        # the return type
        self.returnTech = chebtec

        # the projection
        self.projection = None

        # the default change of basis matrix
        self.S0 = None

    @property
    def numConstraints(self):
        return len(self.constraints) + len(self.continuity)

    @property
    def shape(self):
        # TODO: temp for the moment!
        return (self.dimension.squeeze(), self.source.quasiOp.shape[0])

    def getDimAdjust(self):
        """ size of the input space relative to disc.dicretization """
        return self.source.getDimAdjust()

    def getProjOrder(self):
        """ projection order for the rectangualization """
        return self.source.getProjOrder()
        # return np.asarray([self.numConstraints])

    def bc_rows(self, B):
        # TODO TEMP: this should all be worked out in the source!
        m = self.shape[1]
        bc_off = B.shape[0]
        bc_per = B.shape[0] // m
        dim = self.dimension - bc_per
        return np.hstack([np.hstack([eqn*bc_per + np.arange(bc_per), bc_off + eqn * dim + np.arange(dim)]) for eqn in range(m)]), bc_per

    def precond(self, *args, **kwargs):
        pass

    def matrix(self, *args, **kwargs):
        """ Creates the discretized approximation of the linear or nonlinear
        operator. Currently this function uses rectangularization to
        impose functional constraints on solutions.
        """
        M, Pc, self.S0 = self.instantiate(self.source.quasiOp, *args, **kwargs)

        # then call reduce to create a rectangular matrix
        PA, P, PS = self.reduce(M, self.S0)

        # Get the constraints
        cols = PA.shape[1]
        B = self.getConstraints(cols)

        # Re-order such that the leading order derivative term is situated on
        # the main matrix diagonal -> Important for easy pre-conditioner
        # construction.
        m = self.shape[1]

        # Assemble the preconditioner - and reduce just like the full matrix
        PC = P * sps.bmat(Pc)

        # If we have boundary conditions change
        if B is not None:
            self.idenRows, bc_per = self.bc_rows(B)

            # Append the functional constraints to the constructed linear operators
            M = sps.vstack((B, PA))
            # print('M = ', M.shape)
            # print('idenRows = ', self.idenRows)
            M = flip_rows(M, self.idenRows)

            PC = sps.vstack([eye(m=bc_per, n=B.shape[1], k=eqn * int(self.dimension))
                             for eqn in range(m)] + [PC])
            PC = flip_rows(PC, self.idenRows).diagonal()
        else:
            # in the case of trig collocation no boundary conditions required.
            M = PA

            # set this to the trivial case
            self.idenRows = np.arange(m * self.dimension)

            # In this case it's very simple
            PC = PC.diagonal()

        # TODO: make this somehow toggleable!
        # print('PA = ')
        # print(PA.todense())
        # print('Stacked!')
        # print(M.todense())
        # print('det(M) = ', LA.det(M.todense()))
        # print('rank(M) = ', np.linalg.matrix_rank(M.todense()))

        # Compute the pre-conditioner!
        if np.any((np.abs(np.real(PC)) < 1e-8) & (np.abs(np.imag(PC)) < 1e-8)):
            # The pre-conditioner has a zero diagonal!
            self.Pc = None
        else:
            self.Pc = sps.spdiags(1. / PC, 0, PC.size, PC.size)

        # Store the basic basis conversion matrix
        self.S0 = sps.bmat(self.S0)

        # store the projection matrix
        self.P = P

        # create the main projection operator
        self.projection = RectangularProjection(self.P, self.S0)

        # create the block matrix for S
        return M.tobsr(), self.Pc, self.S0

    def matrix_moore(self, *args, **kwargs):
        """ TODO: this really should not be here. FIXME in the future! """
        M, _, S0 = self.instantiate(self.source.quasiOp_bif, *args, **kwargs)
        M = blockmat(M)

    def matrix_inverse_basis(self, *args, **kwargs):
        S = self.instantiate_i(self.source.quasiOp, *args, **kwargs)
        return sps.block_diag(S)

    def matrix_full(self, *args, **kwargs):
        M, _, S0 = self.instantiate(self.source.quasiOp, *args, **kwargs)
        M = blockmat(M)

        # Get the inverse basis transformation
        S = self.instantiate_i(self.source.quasiOp, *args, **kwargs)
        M = sps.block_diag(S) * M

        # create the block matrix for S
        return M.tocsr()

    def matrix_adjoint(self, bc=False, invert=True, *args, **kwargs):
        """ Creates the discretized approximation of the adjoint linear operator.
        Currently this function uses rectangularization to impose functional constraints on solutions.
        """
        M, S0 = self.instantiate_adjoint(self.source.quasiOp, *args, **kwargs)
        M = blockmat(M)

        # Get the inverse basis transformation
        if invert:
            S = self.instantiate_i(self.source.quasiOp, *args, **kwargs)
            M = sps.block_diag(S) * M

        if bc:
            # then call reduce to create a rectangular matrix
            PA, P, PS = self.reduce(M, S0)

            # Get the constraints
            cols = PA.shape[1]
            B = self.getConstraints(cols)
            self.idenRows, _ = self.bc_rows(B)

            # Append the functional constraints to the constructed linear operators
            M = sps.vstack((B, PA))
            M = flip_rows(M, self.idenRows)

        # create the block matrix for S
        return M.tobsr()

    def matrix_nonproj(self, *args, **kwargs):
        M, Pc, S0 = self.instantiate(self.source.quasiOp, *args, **kwargs)

        # then call reduce to create a rectangular matrix
        PA, P, PS = self.reduce(M, S0)

        # Get the constraints
        cols = PA.shape[1]
        B = self.getConstraints(cols)

        # Re-order such that the leading order derivative term is situated on
        # the main matrix diagonal -> Important for easy pre-conditioner
        # construction.
        m = self.source.linOp.shape[1]
        # TODO TEMP: this should all be worked out in the source!
        bc_off = B.shape[0]
        bc_per = B.shape[0] // m
        dim = self.dimension - bc_per
        self.idenRows = np.hstack([np.hstack(
            [eqn*bc_per + np.arange(bc_per), bc_off + eqn * dim + np.arange(dim)]) for eqn in range(m)])

        # Append the functional constraints to the constructed linear operators
        Mf = sps.vstack((B, PA))
        Mf = flip_rows(Mf, self.idenRows)

        # Store the basic basis conversion matrix
        S0 = sps.bmat(S0)
        # store the projection matrix
        self.P = P

        # create the block matrix for S
        return sps.bmat(M), Mf.tocsr(), self.P, S0

    def rhs(self, u=None):
        return NotImplemented

    def diff_a(self, u):
        return NotImplemented
