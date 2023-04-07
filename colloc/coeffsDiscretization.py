#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
import itertools
from copy import copy

try:
    from scipy.sparse import csr_array
except ImportError:
    from scipy.sparse import csr_matrix as csr_array

import scipy.sparse.linalg as LAS

from fun import Fun
from functional import Functional
from colloc.OpDiscretization import OpDiscretization
from colloc.chebcolloc.chebcolloc2 import chebcolloc2
from cheb.detail import polyval, polyfit
from cheb.chebpts import quadwts


class coeffsDiscretization(OpDiscretization):
    """ Abstract class for coefficient-based discretization of operators. """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coeffs = kwargs.pop('coeffs', np.zeros(1))
        self.coeffs_orig = copy(self.coeffs)

    def getCoeffs(self):
        return self.coeffs

    def getOutputSpace(self):
        #return max(0, self.coeffs.shape[0] - 1)
        # This should return the
        return max(0, self.source.getOutputSpace())

    def quasi2diffmat(self, source, *args, **kwargs):
        return NotImplemented

    def quasi2precond(self, source, *args, **kwargs):
        return NotImplemented

    def instantiate_i(self, source, *args, **kwargs):
        n, m = source.shape
        M = np.empty(n, dtype=object)

        # Currently only has support for one ODE - nothing fancy yet
        for i in range(n):
            M[i] = self.iconvert(0)

        return M

    def instantiate(self, source, precond=False, adjoint=False, *args, **kwargs):
        # Grab the dimension
        dim = self.dimension[0]

        # Move this somewhere else
        M = np.empty(source.shape, dtype=object)
        P = np.empty(source.shape, dtype=object) if precond else None
        S = np.empty(source.shape, dtype=object)

        # Currently only has support for one ODE - nothing fancy yet
        for i, j in np.ndindex(source.shape):
            M[i, j], S[i, j] = self.quasi2diffmat(source[j, i] if adjoint else source[i, j],
                                                  basis_conv=(i == j),
                                                  adjoint=adjoint, *args, **kwargs)

            # If we are not construction the precond we are done
            if not precond: continue

            # Only generate a preconditioner for the diagonal components
            P[i, j] = self.quasi2precond(source[i, j]) if i == j else csr_array((dim, dim))

        return M, P, S

    def instantiate_c(self, *args, **kwargs):
        # TODO: merge with above function!
        # Move this somewhere else
        c_shape = self.source.cmat.shape
        M = np.empty(c_shape, dtype=object)
        # P = np.empty(c_shape, dtype=object)

        # Currently only has support for one ODE - nothing fancy yet
        n, m = c_shape
        for i, j in itertools.product(range(n), range(m)):
            M[i, j] = self.quasi2cmat(self.source.cmat[i, j], basis_conv=(i == j), *args, **kwargs)

            # only generate a preconditioner for the diagonal components
            #if i == j:
            #    P[i, j] = self.quasi2precond(self.source[i, j])
            #else:
            #    dim = self.dimension[0]
            #    P[i, j] = csr_matrix((dim, dim))

        return M, None

    def getConstraints(self, n):
        """
            Get constraints and continuity of a linear operator.

            - Returns: a matrix discretization of the constraints and
            continuity conditions from the linear operator in Disc.source
        """
        blocks = self.convertFunctional(n)

        if blocks is None:
            return None

        nrows = blocks.shape[0]
        ncols = blocks[0].shape[1]

        B = np.zeros((nrows, ncols))
        for i, block in enumerate(blocks):
            B[i, :] = block

        return B

    def convertFunctional(self, n):
        """ We can't represent functional blocks via coeffs.

        Currently this assumes that item is the matrix representation of the
        functional with respect to a value point collocation.
        """
        if not self.constraints:
            return None

        # number of constraints
        nc = len(self.constraints)

        # number of equations
        n, m = self.shape

        blocks = np.empty(nc, dtype=object)
        if self.constraints[0].compiled is not None:
            for i, constraint in enumerate(self.constraints):
                cc = constraint.compiled
                blocks[i] = np.zeros((1, n * m))
                j = cc.shape[1] // m

                # Need to apply the Fourier transform trick per equation element!
                for k in range(m):
                    blocks[i][:, k*n:(k+1)*n] = cc[:, k*j:k*j+n]

        else:
            # Generate a value collocation to first represent the functionals
            fun = Fun(op=m * [lambda x: np.zeros_like(x)], type='cheb').prolong(n)
            valColloc = chebcolloc2(self.source, values=fun.values, domain=self.domain)

            # get the constraint matrices
            vConstraints = valColloc.getConstraints(n)

            # convert from value-space to coeff-space using polyval
            for i, constraint in enumerate(vConstraints):
                blocks[i] = np.zeros_like(constraint)

                # Need to apply the Fourier transform trick per equation element!
                for k in range(m):
                    Ml = np.rot90(constraint[:, k*n:(k+1)*n]).astype(np.double)
                    blocks[i][:, k*n:(k+1)*n] = np.rot90(polyval(Ml), -1)

        return blocks

    def convertToFunctional(self, f):
        func = Functional(f, n=self.source.dshape[0]-2)
        return np.asarray(func).squeeze()

    def toFunction(self, functionals):
        N, M = functionals.shape
        coeffs = np.empty((M, N), dtype=float, order='F')
        rescaleFactor = 0.5 * np.diff(self.domain)
        n, m = self.shape
        w = quadwts(n)

        for k, functional in enumerate(functionals):
            functional = functional.reshape((m, n))

            v = np.empty((n, m), dtype=float)
            for i, row in enumerate(functional):
                row = np.atleast_2d(row)
                v[:, i, None] = np.rot90(polyfit(np.rot90(row)).T / w) / rescaleFactor

            fun = Fun.from_values(v, m, domain=self.domain)
            coeffs[:, k] = self.toFunctionOut(fun.coeffs).flatten(order='F')

        return coeffs
