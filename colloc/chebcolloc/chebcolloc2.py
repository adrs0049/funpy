#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as onp
import jax.numpy as np
import scipy.linalg as LA
from scipy.sparse import eye, csr_matrix, bmat
from sparse.csr import delete_rows_csr, eliminate_zeros_csr

from colloc.valsDiscretization import valsDiscretization
from colloc.chebcolloc.baryDiffMat import diffmat
from cheb.chebpy import chebtec
from cheb.chebpts import chebpts_type1, chebpts_type2, barymat

class chebcolloc2(valsDiscretization):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        return chebcolloc2(values=self.values[:, key], domain=self.domain)

    @property
    def tech(self):
        return chebtec

    def toValues(self):
        " Converts a chebtech2 to values at 2nd kind points"""
        pass

    def eval(self, locations):
        """ Evaluation functional for chebcolloc

            return a functional that evaluates the chebyshev polynomial
            represented by a colloc directization at the given point loc
        """
        locations = np.atleast_1d(locations)
        assert locations >= self.domain[0] and locations <= self.domain[1], \
                'Evaluation point %.4g must be inside the domain %s!' % (locations, self.domain)

        # find the collocation points and create an empty functional
        x, _, v, _ = self.functionPoints()
        return barymat(locations, x, v)

    def functionPoints(self):
        return self.points(lambda N: chebpts_type2(N, interval=self.domain))

    def equationPoints(self):
        return self.points(lambda N: chebpts_type1(N, interval=self.domain))

    def diff(self, k=1, axis=0, *args, **unused_kwargs):
        # TODO: Don't call this directly use the one provided by the child!
        length = self.domain[1] - self.domain[0]
        # Can't call self diffmat as it currently makes JAX unhappy!
        mat = diffmat(self.x, k=k) # * (2/length)**k
        return chebcolloc2(values=np.dot(mat, self.values), domain=self.domain)

    def diffmat(self, k=1, axis=0, **unused_kwargs):
        domain = self.domain
        n = self.dimension
        if k == 0:
            return eye(onp.sum(n))

        # assuming that we only have on interval
        blocks = onp.empty(self.numIntervals, dtype=object)
        for i in range(self.numIntervals):
            length = domain[i+1] - domain[i]
            blocks[i] = diffmat(self.x, k=k) * (2/length)**k
        return LA.block_diag(blocks)
