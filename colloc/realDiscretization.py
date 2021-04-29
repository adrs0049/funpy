#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
from scipy.sparse import csr_matrix
from copy import copy
import itertools

from fun import Fun
from colloc.OpDiscretization import OpDiscretization


class realDiscretization(OpDiscretization):
    """
        Simple discretization of R^n.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'real'
        self.constraints = []

    def quasi2diffmat(self, source, *args, **kwargs):
        #print('coeffs = ', source.getCoeffs()[0].coeffs)
        c_c = np.asarray(source.getCoeffs()[0]).item()

        icoeffs = source.getICoeffs()
        integrand = source.getIntegrand()
        for i in range(source.numNonLocal):
            c_c += np.asarray(icoeffs[i]).item() * np.asarray(integrand[i]).item()

        return c_c

    def instantiate(self, source, *args, **kwargs):
        # Move this somewhere else
        M = np.empty(source.shape, dtype=np.float)

        # Currently only has support for one ODE - nothing fancy yet
        n, m = source.shape
        for i, j in itertools.product(range(n), range(m)):
            M[i, j] = self.quasi2diffmat(source[i, j], *args, **kwargs)

        return M

    def matrix(self, *args, **kwargs):
        n, m = self.source.linOp.shape
        self.projection = np.eye(n)
        M = self.instantiate(self.source.quasiOp)
        self.S0 = np.eye(n)   # Change of basis matrix
        self.P = np.eye(n)    # Projection matrix
        return csr_matrix(M), np.eye(n), np.eye(n)

    def matrix_full(self, *args, **kwargs):
        M, _, _ = self.matrix(*args, **kwargs)
        return M

    def matrix_adjoint(self, *args, **kwargs):
        M, _, _ = self.matrix(*args, **kwargs)
        return M.T

    def rhs_detail(self, u=None):
        return self.rhs(u=u)

    def rhs(self, u=None):
        if u is not None:
            assert np.all(self.domain == u.domain), 'Domain mismatch %s != %s!' % (self.domain, u.domain)
            self.source.update_partial(u)

        return self.source.rhs.values.flatten(order='F')

    def diff_a(self, u=None, *args, **kwargs):
        dps = self.source.pDer(u, *args, **kwargs)
        arr = np.hstack([dp.prolong(self.shape[0]).coeffs for dp in dps]).flatten(order='F').squeeze()
        return arr

    def diff_a_detail(self, *args, **kwargs):
        return self.diff_a(*args, **kwargs)

    def project_vector_cts(self, vector):
        return vector
