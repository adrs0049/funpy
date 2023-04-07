#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
from pylops import LinearOperator
import numpy as np
import scipy.sparse.linalg as LAS
import scipy.linalg as LA

from fun import Fun, h1norm, norm, norm2
from support.Functional import Functional
from cheb.chebpts import quadwts
from cheb.diff import computeDerCoeffs


class FrechetDeflatedPrecond(LinearOperator):
    def __init__(self, Pf, b, dtype=None, *args, **kwargs):
        self.Pf = Pf
        self.PfT = None
        self.b = b

        self.shape = self.Pf.shape
        self.dtype = np.dtype(dtype)
        self.explicit = False

        self.ks = kwargs.get('ks', 0)
        self.eta = kwargs.get('eta', 1.0)

        # Pre-compute this!
        self.functional = kwargs.get('functional', None)

        if self.functional is not None:
            self.denom = self.__compute_denom()

    def __compute_denom(self):
        # compute the values for the inner-product
        return self.eta**2 + self.functional(self.b)

    def _matvec(self, x):
        """
        Computes the action of the inverse of

            P_G = eta P_F + F outer d

            which is computed via the Shermann-Morrisson formula
        """
        if self.ks == 0:
            return self.Pf.solve(x)

        term1 = self.Pf.solve(x)
        term2 = self.b * (self.functional(term1) / self.denom)
        return (term1 - term2) / self.eta

    def _rmatvec(self, x):
        if self.PfT is None:
            self.PfT = self.Pf.transpose()

        if self.ks == 0:
            return self.PfT.dot(x)

        assert False, 'FIXME'

    def to_matrix(self):
        return self.tosparse()
