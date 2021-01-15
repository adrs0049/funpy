#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
import scipy.linalg as LA
import scipy.sparse as sps
import scipy.sparse.linalg as LAS
from scipy.sparse import eye, csr_matrix
from sparse.csr import delete_rows_csr, eliminate_zeros_csr

from fun import Fun
from fun import norm

from colloc.coeffsDiscretization import coeffsDiscretization
from colloc.ultraS.matrices import convertmat, convertmat_inv, diffmat, intmat, multmat, delete_rows
from colloc.ultraS.matrices import blockmat
from cheb.chebpts import quadwts
from cheb.detail import polyval


class ultraS(coeffsDiscretization):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO FIXME eventually!
        self.name = 'coeff'

    def convert(self, K1, K2=None):
        n = self.dimension
        if K2 is None:
            K2 = self.getOutputSpace() - 1

        blocks = np.empty(self.numIntervals, dtype=object)
        for i in range(self.numIntervals):
            blocks[i] = convertmat(n[i], K1, K2, format='csr')
        return sps.block_diag(blocks)

    def iconvert(self, K1, K2=None):
        n = self.dimension
        if K2 is None:
            K2 = self.getOutputSpace() - 1

        blocks = np.empty(self.numIntervals, dtype=object)
        for i in range(self.numIntervals):
            blocks[i] = convertmat_inv(n[i], K1, K2, format='csr')
        return sps.block_diag(blocks)

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
            blocks[i] = diffmat(n[i], m=m, format='csr') * (2/length)**m
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

    def mult(self, f, lam):
        n = self.dimension

        blocks = np.empty(self.numIntervals, dtype=object)
        for i in range(self.numIntervals):
            blocks[i] = multmat(n[i], f[i], lam)
        return sps.block_diag(blocks)

    """ Rectangular differentiation matrix support """
    def reduce(self, A, S):
        """
            returns PA, P, PS
        """
        r = self.getProjOrder()
        dim = self.dimension
        dimAdjust = self.getDimAdjust()

        PA = np.empty((1, A.shape[1]), dtype=object)
        P  = np.empty(A.shape[1], dtype=object)
        PS = np.empty((1, A.shape[1]), dtype=object)

        for i in range(A.shape[1]):
            PA[0, i], P[i], PS[0, i] = self.reduceOne(A[:, i], S[:, i], r[i], dim  + dimAdjust[i])

        P  = sps.block_diag(P)
        PA = blockmat(PA)
        PS = blockmat(PS)
        return PA, P, PS

    def reduceOne(self, A, S, m, n):
        """
        Reduces the entries of the column cell arrays A and S from sum(N) x sum(N)
        discretizations to sum(N - M) x sum(N) versions (PA and PS, respectively)
        using the block-projection operator P.

        m: The projection order
        n: dim + dimAdjust

        """
        # Projection matrix for US remove the last m coeffs
        P = eye(np.sum(n)).tocsr()
        nn = np.cumsum(np.hstack((0, n)))
        n = np.asarray([n])
        # v are the row indices which are to be removed by projection
        v = np.empty(0)
        v = np.hstack((v, nn[0] + n[0] - np.arange(1, m + 1))).astype(int)
        P = delete_rows_csr(P, v)

        # project each component of A and S:
        PA = np.copy(A)
        PS = np.copy(S)

        for j in range(PA.size):
            if P.shape[1] == A[j].shape[0]:
                PA[j] = delete_rows(PA[j], v)
                PS[j] = delete_rows(PS[j], v)
            else:
                PA[j] = A[j]
                PS[j] = S[j]

        # TODO: can we deal with those expansion in a better way?
        PA = blockmat(np.expand_dims(PA, axis=1))
        PS = blockmat(np.expand_dims(PS, axis=1))

        # Don't want to project scalars!
        if m == 0 and A[0].shape[1] < np.sum(n):
            P = eye(A.shape[1])

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
            # S = self.convert(j)
            # D = self.diff(j)
            # M = self.mult(coeff[j], j)
            # print('C[%d]:' % j, coeff[j])
            # print('S[%d]:' % j, S.todense())
            # print('D[%d]:' % j, D.todense())
            # print('M[%d]:' % j, M.todense())
            # print('L[%d]:' % j, (self.convert(j) * self.mult(coeff[j], j) * self.diff(j)).todense())
            L += self.convert(j) * self.mult(coeff[j], j) * self.diff(j)

        # check if we have an integral term defined!
        if info[-1]:
            L += self.convert(0) * self.mult(source.getICoeffs(), 0) * self.int()

        # if transform -> want to make sure everything is with respect to standard basis!
        transform = kwargs.get('transform', False)
        if transform:
            Sinv = self.iconvert(0)
            L = Sinv * L

        # must remove almost zero elements now
        L = eliminate_zeros_csr(L)

        # create the conversion matrix as well
        S = self.convert(0) if basis_conv else csr_matrix(L.shape)
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
        Lt = csr_matrix((np.sum(dim), np.sum(dim)))

        # check if we have an integral term defined!
        if info[-1]:
            Lt += self.convert(0) * self.mult(source.getICoeffs(), 0) * self.int()

        # cut off some rows
        L = csr_matrix((np.sum(dim), np.sum(dim))) if source.diff_order >= 0 else csr_matrix((1, np.sum(dim)))
        L = Lt[0, :]

        # must remove almost zero elements now
        L = eliminate_zeros_csr(L)

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
        P = csr_matrix((np.sum(dim), np.sum(dim)))
        M = self.mult(coeff[-1], coeff.shape[0]-1)
        if np.any(M.diagonal() == 0):
            P = self.diff(coeff.shape[0]-1).tocsr()
        else:
            P = M * self.diff(coeff.shape[0]-1).tocsr()

        # must remove almost zero elements now
        P = eliminate_zeros_csr(P)
        return P

    def rhs(self, u=None):
        """ Generate the discretization of the right hand side """
        # If u is not None -> update the source first!
        if u is not None:
            assert np.all(self.domain == u.domain), 'Domain mismatch %s != %s!' % (self.domain, u.domain)
            self.source.update_partial(u)

        # now project and apply change of basis matrix!
        return self.project_vector_cts(self.source.rhs.values.flatten(order='F'))

    def project_vector_cts(self, vector):
        # now project and apply change of basis matrix!
        # TODO: why does numpy sometimes up-cast to complex here?
        arr = np.real(self.projection._matvec(vector))
        return np.hstack((self.source.cts_res.flatten(), arr))[self.idenRows]

    def project_vector(self, vector):
        """
        Applies the projection and change of basis to a vector.
        For instance the vector representing the derivative of the nonlinear
        function with respect to a model parameter.

        The rows corresponding to the boundary conditions are replaced by zero.
        """
        arr = self.projection._matvec(vector)
        return np.concatenate((np.zeros(2 * self.source.dshape[1]), arr))[self.idenRows]

    def diff_a_detail(self, u, *args, **kwargs):
        # Returns a function object
        dps = self.source.pDer(u, *args, **kwargs)
        return np.hstack([dp.prolong(self.shape[0]).coeffs for dp in dps]).flatten(order='F').squeeze()

    def diff_a(self, u, *args, **kwargs):
        arr = self.diff_a_detail(u, *args, **kwargs)
        return self.project_vector(arr)

    def diff_a_adjoint(self, u):
        # Sigh need this as a chebfun
        dps = self.source.pDer(u)

        n = int(self.shape[0])
        w = quadwts(n)
        rescaleFactor = 0.5 * np.diff(self.domain)

        rhs = np.empty(len(dps), dtype=object)
        for m in range(rhs.size):
            rhs[m] = rescaleFactor * np.rot90(polyval(np.rot90(w * dps[m].prolong(n).values.T)), -1)

        return np.hstack(rhs)
