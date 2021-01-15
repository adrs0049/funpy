#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
from scipy.sparse import csr_matrix
from copy import copy
import itertools

from funpy.fun import Fun
from funpy.colloc.OpDiscretization import OpDiscretization


class realDiscretization(OpDiscretization):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'real'
        self.constraints = []

    def quasi2diffmat(self, source, *args, **kwargs):
        c_c = source.getCoeffs()[0]
        c_i = source.getICoeffs()
        # print('c_c:', np.asarray(c_c), ' c_i:', np.asarray(c_i))
        return np.asarray(c_c).item() + np.asarray(c_i).item()

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
        M = self.instantiate(self.source.quasiOp)
        self.S0 = np.eye(n)
        return csr_matrix(M), np.eye(n), np.eye(n)

    def projection(self):
        return self.S0

    def rhs(self, u=None):
        if u is not None:
            assert np.all(self.domain == u.domain), 'Domain mismatch %s != %s!' % (self.domain, u.domain)
            self.source.update_partial(u)

        return self.source.rhs.values.flatten(order='F')

    def diff_a(self, u=None):
        dps = self.source.pDer(u)
        arr = np.hstack([dp.prolong(self.shape[0]).coeffs for dp in dps]).flatten(order='F').squeeze()
        return arr
