#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
import scipy.sparse as sps
from scipy.sparse import eye, csr_matrix

from funpy import Fun

from ..coeffsDiscretization import coeffsDiscretization
from ..sparse.csr import eliminate_zeros

from .matrices import diffmat, intmat, multmat
from .matrices import blockmat, aggmat


class trigspec(coeffsDiscretization):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'trigspec'

    def diff(self, m):
        """ Generates the differentiation matrix of order m """
        domain = self.domain
        n = self.dimension
        if m == 0:
            return eye(np.sum(n)).tocsr()

        # assuming that we only have on interval
        blocks = np.empty(self.numIntervals, dtype=object)
        for i in range(self.numIntervals):
            length = domain[i+1] - domain[i]
            blocks[i] = diffmat(n[i], m=m) * (2 * np.pi / length)**m
        return sps.block_diag(blocks)

    def int(self):
        domain = self.domain
        n = self.dimension
        # assuming that we only have on interval
        blocks = np.empty(self.numIntervals, dtype=object)
        for i in range(self.numIntervals):
            length = domain[i+1] - domain[i]
            blocks[i] = intmat(n[i], format='csr') * 0.5 * length
        return sps.block_diag(blocks)

    def mult(self, f):
        """ Creates the multiplication matrix! f -> must be a trigtech object here """
        n = self.dimension

        blocks = np.empty(self.numIntervals, dtype=object)
        for i in range(self.numIntervals):
            blocks[i] = multmat(n[i], f[i])
        return sps.block_diag(blocks)

    def aggregation(self, m=0):
        """ Generates the non-local aggregation term

            m -> order of the derivative in front of the aggregation term.

            Notes: Currently only implements a known simple integration kernel.
            Future work is to extend this!
        """
        domain = self.domain
        n = self.dimension

        blocks = np.empty(self.numIntervals, dtype=object)
        for i in range(self.numIntervals):
            rescaleFactor = 0.5 * (domain[i + 1] - domain[i])
            blocks[i] = aggmat(n[i], m=m, rescaleFactor=1./rescaleFactor, format='csr')
        return sps.block_diag(blocks)

    """ Rectangular differentiation matrix support """
    def reduce(self, A, S):
        """ Dimension reduction for the operator -> not required for periodic problems """
        PA = blockmat(A)
        P = eye(PA.shape[0])
        PS = P
        return PA, P, PS

    def quasi2diffmat(self, source, basis_conv=True, *args, **kwargs):
        """ Converts the coefficients of the linear differential operator into
            a matrix representation of the linear differential operator "M".

            It also returns the matrix S -> which is the basis conversion
            matrix from C^{0} -> C^{m} where m is the order of the highest
            order differential operator present.
        """
        dim = self.dimension
        # get operator coefficients
        coeff = source.getCoeffs()
        # Info is sorted by differential term order from 0 -> diffOrder
        info = source.info()

        L = csr_matrix((np.sum(dim), np.sum(dim)))
        for j in range(coeff.shape[0]):
            # Only call the various matrix assembly functions only if the terms are non-zero!
            if not info[j]: continue

            # form the D^(j - 1) term
            #M = self.mult(coeff[j])
            #D = self.diff(j)
            #print('coeff[%d] = ' % j, coeff[j])
            #print('M[%d] = ' % j, M.shape)
            #print('D[%d] = ' % j, D.shape)
            L += self.mult(coeff[j]) * self.diff(j)

        # check if we have an integral term defined!
        if info[-1]:
            L += self.mult(source.getICoeffs()) * self.int()

        # Deal with aggregation terms
        coeff = source.getNlCoeffs()
        info = source.nl_info()
        for j in range(coeff.shape[0]):
            # Only call the various matrix assembly functions only if the terms are non-zero!
            if not info[j]: continue

            # form the D^(j - 1) term
            L += self.mult(coeff[j]) * self.diff(j) * self.aggregation()

        # must remove almost zero elements now
        L = eliminate_zeros(L)

        # create the conversion matrix as well
        S = csr_matrix(L.shape)
        return L, S

    def quasi2precond(self, source):
        """ Converts the coefficients to a simple pre-conditioner """
        dim = self.dimension
        # Info is sorted by differential term order from 0 -> diffOrder
        info = source.info()

        # get operator coefficients
        coeff = source.getCoeffs()
        if coeff.size == 0:
            return NotImplemented

        # Take the diagonal of the matrix + the multiplication operator of the coefficient!
        P = csr_matrix((np.sum(dim), np.sum(dim)))
        for j in range(coeff.shape[0]):
            if not info[j]: continue
            P += self.mult(coeff[j]) * self.diff(j)

        # Let's just use the diagonal here!
        P = sps.spdiags(P.diagonal(), 0, P.shape[0], P.shape[1])

        return P

    # def rhs(self, u=None):
    #     """ Generate the discretization of the right hand side """
    #     # If u is not None -> update the source first!
    #     if u is not None:
    #         assert np.all(self.domain == u.domain), 'Domain mismatch %s != %s!' % (self.domain, u.domain)
    #         self.source.update_partial(u)

    #     # Trivial for trig collocations
    #     return self.source.rhs.values.flatten(order='F')
    # def projection(self):
    #     n = self.dimension
    #     return eye(np.sum(n)).tocsr()

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
