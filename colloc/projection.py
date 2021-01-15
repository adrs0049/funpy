#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
from sparse.csr import csr_vappend
from pylops import LinearOperator


class RectangularProjection(LinearOperator):
    """
        Implements an operator for the rectangular projections used
        in collocating differential operators.
    """
    def __init__(self, P, S0, dtype=None, *args, **kwargs):
        # The projection matrix
        self.op = P * S0

        # Basic setup for the linear operator
        self.shape = self.op.shape
        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        return self.op.dot(x).squeeze()

    def _rmatvec(self, x):
        """ Implements x = A^H y """
        assert False, ''
        return np.zeros_like(x)
