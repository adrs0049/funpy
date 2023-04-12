#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
import scipy.sparse as sps
from scipy.sparse import eye

try:
    from scipy.sparse import csr_array, csc_array
except ImportError:
    from scipy.sparse import csr_matrix as csr_array
    from scipy.sparse import csc_matrix as csc_array

from ..sparse.csr import delete_rows_csr, eliminate_zeros

from funpy import Fun, ultra2ultra
from funpy.cheb import quadwts
from funpy.cheb import polyval

from colloc.coeffsDiscretization import coeffsDiscretization

from .matrices import convertmat, convertmat_inv, diffmat, multmat
from .matrices import blockmat, realmat, intmat, zeromat
from .matrices import reduceOne


class ultraS(coeffsDiscretization):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO FIXME eventually!
        self.name = 'coeff'

    def convert(self, K1, K2=None, format='csr'):
        n = self.dimension
        if K2 is None:
            K2 = self.getOutputSpace() - 1

        blocks = np.empty(self.numIntervals, dtype=object)
        for i in range(self.numIntervals):
            blocks[i] = convertmat(n[i], K1, K2, format=format)

        return sps.block_diag(blocks, format=format)

    def iconvert(self, K1, K2=None, format='csr'):
        n = self.dimension
        if K2 is None:
            K2 = self.getOutputSpace() - 1

        blocks = np.empty(self.numIntervals, dtype=object)
        for i in range(self.numIntervals):
            blocks[i] = convertmat_inv(n[i], K1, K2, format=format)

        return sps.block_diag(blocks, format=format)

    def diff(self, m, format='csr'):
        """ Generates the differentiation matrix of order m """
        domain = self.domain
        n = self.dimension
        if m == 0:
            return eye(np.sum(n), format=format)

        # assuming that we only have on interval
        blocks = np.empty(self.numIntervals, dtype=object)
        for i in range(self.numIntervals):
            length = domain[i + 1] - domain[i]
            blocks[i] = diffmat(n[i], m=m, format=format) * (2 / length)**m

        return sps.block_diag(blocks, format=format)

    def int(self, format='csr'):
        domain = self.domain
        n = self.dimension
        # assuming that we only have on interval
        blocks = np.empty(self.numIntervals, dtype=object)
        for i in range(self.numIntervals):
            length = domain[i + 1] - domain[i]
            blocks[i] = intmat(n[i], format='csr') * 0.5 * length

        return sps.block_diag(blocks, format=format)

    def real(self, format='csr'):
        # domain = self.domain
        n = self.dimension
        blocks = np.empty(self.numIntervals, dtype=object)
        for i in range(self.numIntervals):
            # length = domain[i+1] - domain[i]
            blocks[i] = realmat(n[i], format=format)

        return sps.block_diag(blocks, format=format)

    def mult(self, f, lam, **kwargs):
        n = self.dimension

        blocks = np.empty(self.numIntervals, dtype=object)
        for i in range(self.numIntervals):
            blocks[i] = multmat(n[i], f[i], lam, **kwargs)
        return sps.block_diag(blocks)

    def reduce(self, A, S, adjoint=False):
        """
        Project the matrices A and S to the appropriate
        rectangular differentiation matrices.

        Returns:
            - PA: Rectangular version of A.
            - P:  Projection matrix.
            - PS: Rectangular version of S.
        """
        r = self.getProjOrder()
        dim = self.dimension
        dimAdjust = self.getDimAdjust()

        PA = np.empty((1, A.shape[1]), dtype=object)
        P  = np.empty(A.shape[1], dtype=object)
        PS = np.empty((1, A.shape[1]), dtype=object)

        for i in range(A.shape[1]):
            PA[0, i], P[i], PS[0, i] = reduceOne(A[:, i], S[:, i],
                                                 r[i], dim  + dimAdjust[i])
                                                 #adjoint=adjoint)

        P  = sps.block_diag(P)
        PA = sps.bmat(PA)
        PS = sps.bmat(PS)
        return PA, P, PS

    def quasi2diffmat(self, source, basis_conv=True, adjoint=False, format='csr', *args, **kwargs):
        """ Converts the coefficients of the linear differential operator into
            a matrix representation of the linear differential operator "M".

            It also returns the matrix S -> which is the basis conversion
            matrix from C^{0} -> C^{m} where m is the order of the highest
            order differential operator present.

            Discretize a sequence of linear differential operators of the forms:

                                    du^N
                L[u] := a^N(x, u)  ------
                                    dx^N

            The result must output in the C^{n} basis, thus in addition we multiply
            with the required basis conversion matrix.

            Adjoint generation: Given the general second order operator

                              du^2          du
                L[u] := f(x) ------ + g(x) ---- + h(x) u
                              dx^2          dx

            The adjoint operator is given by

                              du^2               du
               L*[u] := f(x) ------ + (2f' - g) ---- + (f'' - g' + h) u
                              dx^2               dx

            For the moment we assume that the operator blocks are self-adjoint i.e.
            can be written in the form:

                        d           du
                L[u] = ---- ( f(x) ---- ) + h(x) u
                        dx          dx
        """
        dim = self.dimension
        # get operator coefficients
        coeff = source.getCoeffs()
        # Info is sorted by differential term order from 0 -> diffOrder
        info = source.info()

        # Create sparse matrix object.
        L = zeromat((np.sum(dim), np.sum(dim)), format=format)

        for j in range(coeff.shape[0]):
            # Only call the various matrix assembly functions only if the terms are non-zero!
            if not info[j]: continue

            # form the D^(j - 1) term
            #   => If adjoint and odd => put in a negative
            if adjoint and j & 1:
                L -= self.convert(j, format=format) * self.mult(coeff[j], j, format=format) * self.diff(j, format=format)
            #   => If not an adjoint simply construct.
            else:
                L += self.convert(j, format=format) * self.mult(coeff[j], j, format=format) * self.diff(j, format=format)

        # check if we have an integral term defined!
        if info[-1]:
            """
            Discretize a sequence of non-local operators of the forms:

                L[u] := c(x, u) ∫ d(x, u) u(x) dx

            The adjoint operator is given by

                M[u] := d(x, u) ∫ c(x, u) u(x) dx

            """
            # We might have a sequence of such operators
            if adjoint:
                icoeffs = source.getIntegrand()
                integrands = source.getICoeffs()
            else:
                icoeffs = source.getICoeffs()
                integrands = source.getIntegrand()

            for j in range(icoeffs.shape[0]):
                L += self.convert(0, format=format) * self.mult(icoeffs[j], 0, format=format) * self.int(format=format) * self.mult(integrands[j], 0, format=format)

        # if transform -> want to make sure everything is with respect to standard basis!
        # transform = kwargs.get('transform', False)
        # if transform:
        #     Sinv = self.iconvert(0, format=format)
        #     L = Sinv * L

        # must remove almost zero elements now
        #L = eliminate_zeros(L)

        # create the conversion matrix as well
        if adjoint:
            S = self.iconvert(0, format=format) if basis_conv else zeromat(L.shape, format=format)
        else:
            S = self.convert(0, format=format) if basis_conv else zeromat(L.shape, format=format)

        return L, S

    def quasi2cmat(self, source, basis_conv=True, *args, **kwargs):
        """ Converts the coefficients of the linear differential operator into
            a matrix representation of the linear differential operator "M".

            It also returns the matrix S -> which is the basis conversion
            matrix from C^{0} -> C^{m} where m is the order of the highest
            order differential operator present.
        """
        dim = self.dimension
        # Info is sorted by differential term order from 0 -> diffOrder
        info = source.info()

        # Create matrix
        Lt = csr_array((np.sum(dim), np.sum(dim)))

        # check if we have an integral term defined!
        if info[-1]:
            Lt += self.convert(0) * self.mult(source.getICoeffs(), 0) * self.int()

        # cut off some rows
        L = csr_array((np.sum(dim), np.sum(dim))) if source.diff_order >= 0 else csr_array((1, np.sum(dim)))
        L = Lt[0, :]

        # must remove almost zero elements now
        #L = eliminate_zeros(L)

        # create the conversion matrix as well
        return L

    def quasi2precond(self, source):
        """ Converts the coefficients to a simple pre-conditioner """
        dim = self.dimension
        # get operator coefficients
        coeff = source.getCoeffs()
        if coeff.size == 0:
            return NotImplemented

        # Take the diagonal of the matrix + the multiplication operator of the coefficient!
        P = csr_array((np.sum(dim), np.sum(dim)))
        M = self.mult(coeff[-1], coeff.shape[0] - 1)
        if np.any(M.diagonal() == 0):
            P = self.diff(coeff.shape[0] - 1).tocsr()
        else:
            P = M * self.diff(coeff.shape[0] - 1).tocsr()

        return P

    def project_vector(self, vector, basis=True, cts=False):
        """
        Applies the projection and change of basis to a vector.
        For instance the vector representing the derivative of the nonlinear
        function with respect to a model parameter.

        The rows corresponding to the boundary conditions are replaced by zero.

        Arguments:

            vector: The vector that will the projected.
            basis:  When True also apply the basis transformation.
            cts:    Append the constraints residuals to the top. Otherwise set to zero.

        """
        if isinstance(vector, Fun):
            vector = vector.prolong(self.source.dshape[0]).flatten()

        if basis:
            arr = np.atleast_1d(self.projection._matvec(vector))
        else:
            arr = np.atleast_1d(self.proj._matvec(vector))

        carr = self.source.cts_res.flatten() if cts else np.zeros(self.numConstraints)
        return np.concatenate((carr, arr))

    def toFunctionOut(self, coeffs):
        """ Convert function in Chebyshev basis into the output space """
        return ultra2ultra(coeffs, 0, self.getOutputSpace())

    def toFunctionIn(self, coeffs, *args, **kwargs):
        """
            Convert function from the output basis i.e.

                Function output in R x Function space.

            into a Chebyshev polynomial
        """
        n = self.numConstraints
        m = self.shape[1]
        lam_in = kwargs.pop('lam_in', self.getOutputSpace())
        return (np.asarray(coeffs[:n]),
                Fun.from_ultra(coeffs[n:], m, lam_in, domain=self.domain))
