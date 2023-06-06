#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
from pylops import LinearOperator


class RectangularProjection(LinearOperator):
    """
        Implements an operator for the rectangular projections used
        in collocating differential operators.
    """
    def __init__(self, matrix, adjoint=None, dtype=None, *args, **kwargs):
        # The projection matrix
        self.mat = matrix
        self.adj = adjoint

        # Basic setup for the linear operator
        self.shape = self.mat.shape
        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        """
            Implements y = A x
        """
        return self.mat.dot(x).squeeze()

    def _rmatvec(self, y):
        """
            Implements x = A^H y
        """
        return self.adj.dot(y).squeeze()
