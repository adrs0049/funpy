#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import itertools
import warnings
import numpy as np
import scipy as sp
import scipy.sparse as sps
import scipy.linalg as LA
import scipy.sparse.linalg as LAS
from scipy.sparse import eye, bmat

try:
    from scipy.sparse import csr_array, csc_array
except ImportError:
    from scipy.sparse import csr_matrix as csr_array
    from scipy.sparse import csc_matrix as csc_array

from funpy import Fun
from funpy.cheb import chebtech
from funpy import Mapping

from .sparse.csr import flip_rows
from .sparse.csc import flip_cols

from .ultraS.matrices import blockmat
from .projection import RectangularProjection


class OpDiscretization:
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
        return (self.dimension.squeeze().item(), self.source.quasiOp.shape[0])

    def getDimAdjust(self):
        """ size of the input space relative to disc.dicretization """
        return self.source.getDimAdjust()

    def getProjOrder(self):
        """ projection order for the rectangualization """
        return self.source.getProjOrder()

    def matrix(self, *args, **kwargs):
        """ Creates the discretized approximation of the linear or nonlinear
        operator. Currently this function uses rectangularization to
        impose functional constraints on solutions.
        """
        M, Pc, S0 = self.instantiate(self.source.quasiOp, *args, **kwargs)
        PM, P, PS = self.reduce(M, S0)

        # If we have defined constraints add them!
        if self.numConstraints > 0:
            B = self.getConstraints(PM.shape[1])

            # Check that B-rows are not zero
            if np.all(np.sum(np.abs(B), axis=1) == 0.0):
                warnings.warn('OpDiscretization: Boundary condition rows have zero l1 norm!\nMost likely the resulting matrix will be singular! Double check your boundary conditions!')

            # Append the functional constraints to the constructed linear operators
            B  = csr_array(B)
            M  = sps.vstack((B, PM))
        else:
            M = PM

        # Create the projection operator
        self.proj = RectangularProjection(P)
        self.projection = RectangularProjection(PS)

        # create the block matrix for S
        return M, None, self.projection

    def linop(self, *args, **kwargs):
        """ Creates the discretized approximation of the linear or nonlinear
        operator. Currently this function uses rectangularization to
        impose functional constraints on solutions.
        """
        # Create matrix representation of the operator
        M, Pc, S0 = self.instantiate(self.source.quasiOp, *args, **kwargs)
        PM, P, PS = self.reduce(M, S0)

        # Create matrix representation of the operator adjoint
        N, Qc, T0 = self.instantiate(self.source.quasiOp, adjoint=True, format='csr', *args, **kwargs)
        QN, Q, QT = self.reduce(N, T0, adjoint=True)

        # Get the constraints
        B = self.getConstraints(PM.shape[1])

        # Check that B-rows are not zero
        if np.all(np.sum(np.abs(B), axis=1) == 0.0):
            warnings.warn('OpDiscretization: Boundary condition rows have zero l1 norm!\nMost likely the resulting matrix will be singular! Double check your boundary conditions!')

        # Append the functional constraints to the constructed linear operators
        BB = csc_array(self.toFunction(B))
        B  = csr_array(B)
        M  = sps.vstack((B, PM))

        # Add the correct options to the adjoint
        # N = sps.hstack((BB, QN))  # BB here!
        N = sps.vstack((B, QN))

        # Create the projection operator
        self.proj = RectangularProjection(P, Q)
        self.projection = RectangularProjection(PS, QT)

        # create the block matrix for S
        return M, N, self.projection

    def instantiate(self, source, *args, **kwargs):
        return NotImplemented

    def reduce(self, A, S):
        return NotImplemented

    def toFunctionOut(self, coeffs):
        return NotImplemented

    def toFunctionIn(self, coeffs):
        return NotImplemented

    def toValues(self, coeffs):
        return NotImplemented
