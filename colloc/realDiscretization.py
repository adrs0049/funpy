#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np

try:
    from scipy.sparse import csr_array
except ImportError:
    from scipy.sparse import csr_matrix as csr_array

from funpy import Fun

from .OpDiscretization import OpDiscretization
from .projection import RectangularProjection


class realDiscretization(OpDiscretization):
    """
        Simple discretization of R^n.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'real'
        self.constraints = []

    def quasi2diffmat(self, source, *args, **kwargs):
        c_c = np.asarray(source.getCoeffs()[0].prolong(1)).item()

        icoeffs = source.getICoeffs()
        integrand = source.getIntegrand()
        for i in range(source.numNonLocal):
            c_c += np.asarray(icoeffs[i]).item() * np.asarray(integrand[i]).item()

        return c_c

    def instantiate(self, source, *args, **kwargs):
        # Move this somewhere else
        M = np.empty(source.shape, dtype=float)

        # Currently only has support for one ODE - nothing fancy yet
        for i, j in np.ndindex(source.shape):
            M[i, j] = self.quasi2diffmat(source[i, j], *args, **kwargs)

        return M

    def matrix(self, *args, **kwargs):
        n, m = self.source.linOp.shape
        self.projection = np.eye(n)
        M = self.instantiate(self.source.quasiOp)
        self.S0 = np.eye(n)   # Change of basis matrix
        self.P = np.eye(n)    # Projection matrix
        self.proj = RectangularProjection(self.P)
        return csr_array(M), np.eye(n), np.eye(n)

    def linop(self, *args, **kwargs):
        n, m = self.source.linOp.shape
        self.projection = np.eye(n)
        M = self.instantiate(self.source.quasiOp)
        self.S0 = np.eye(n)   # Change of basis matrix
        self.P = np.eye(n)    # Projection matrix
        self.proj = RectangularProjection(self.P)
        return csr_array(M), csr_array(M.T), np.eye(n)

    def project_vector(self, vector, *args, **kwargs):
        return np.asarray(vector)

    def toFunctionOut(self, coeffs):
        return coeffs

    def toFunctionIn(self, coeffs, *args, **kwargs):
        m = self.shape[1]
        return (None, Fun.from_coeffs(coeffs, m, domain=self.domain))
