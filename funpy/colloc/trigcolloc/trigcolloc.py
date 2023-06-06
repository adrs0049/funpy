#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np

import scipy.linalg as LA
import scipy.sparse as sps

from scipy.sparse import eye

from ..fun.trig.trigtech import trigtech
from ..fun.trig.trigpts import trigpts

from ..valsDiscretization import valsDiscretization
from ..chebcolloc.baryDiffMat import diffmat
from ..trigcolloc.matrices import blockmat


class trigcolloc(valsDiscretization):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        return trigcolloc(values=self.values[:, key], domain=self.domain)

    @property
    def tech(self):
        return trigtech

    def toValues(self):
        " Converts a chebtech2 to values at 2nd kind points"""
        pass

    def reduce(self, A, S):
        """ Dimension reduction for the operator -> not required for periodic problems """
        PA = blockmat(A)
        P = eye(A.shape[0])
        PS = P
        return PA, P, PS

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
        return self.points(lambda N: trigpts(N, interval=self.domain))

    def equationPoints(self):
        return self.points(lambda N: trigpts(N, interval=self.domain))

    def diff(self, k=1, axis=0, **unused_kwargs):
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

    def int(self):
        """ The integration operator for a trigonometric collocation """
        domain = self.domain
        n = self.dimension
        # assuming that we only have on interval
        blocks = np.empty(self.numIntervals, dtype=object)
        for i in range(self.numIntervals):
            length = domain[i+1] - domain[i]
            blocks[i] = intmat(n[i]) * 0.5 * length
        return sps.block_diag(blocks)
