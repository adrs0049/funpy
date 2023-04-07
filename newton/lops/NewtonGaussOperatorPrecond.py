#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
import scipy.sparse as sps
from pylops import LinearOperator


class NewtonGaussContinuationCorrectorPrecond(LinearOperator):
    def __init__(self, Du, dtype=None, *args, **kwargs):
        self.Du = Du

        # Basic setup for the linear operator
        self.shape = tuple([sum(x) for x in zip(self.Du.shape, (1, 1))])
        self.dtype = np.dtype(dtype)
        self.explicit = False

        # assemble matrix
        self.mat = sps.lil_matrix(self.shape)
        self.mat[:-1, :-1] = self.Du.to_matrix()
        self.mat[-1, -1] = 1.0
        self.mat = self.mat.tocsr()

    def _matvec(self, x):
        return self.Du._matvec(x[:-1]) + x[-1]

    def to_matrix(self):
        return self.mat
