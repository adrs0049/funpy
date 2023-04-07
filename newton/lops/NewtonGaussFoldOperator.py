#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
import scipy.sparse as sps
from pylops import LinearOperator

from newton.lops.NewtonGaussOperatorPrecond import NewtonGaussContinuationCorrectorPrecond


class NewtonGaussFoldContinuationCorrector(LinearOperator):
    def __init__(self, Du, Da, Db, hxab, dtype=None, *args, **kwargs):
        self.Du = Du.tocsr()
        self.Da = Da
        self.Db = Db
        self.hxab = hxab

        # Basic setup for the linear operator
        self.shape = tuple([sum(x) for x in zip(self.Du.shape, (1, 2))])
        self.dtype = np.dtype(dtype)
        self.explicit = False

        print('shape = ', self.shape)

    def _matvec(self, dx):
        """ Implements y = Ax

            Here the matrix A is given by:

                / D_u F(a, b, u)  D_a F(a, b, u) D_b F(a, b, u) \
                \ D_u h(a, b, u)  D_a h(a, b, u) D_b h(a, b, u) /

            where D_u F(a, b, u) is a n x n matrix:
                  D_a F(a, b, u) is a n x 1 matrix
                  D_b F(a, b, u) is a n x 1 matrix

                  D_u h(a, b, u) is a n x 1 matrix
                  D_a h(a, b, u) is a 1 x 1 matrix
                  D_b h(a, b, u) is a 1 x 1 matrix
        """
        return np.hstack((self.Du @ dx[:-2] + self.Da * dx[-2] + self.Db * dx[-1],
                         np.dot(self.hxab, dx)))

    def _rmatvec(self, dx):
        """ Implements x = A^H y """
        return np.hstack((self.Du._rmatvec(dx), np.dot(self.Da, dx)))

    def to_matrix(self):
        # assemble matrix
        mat = sps.lil_matrix(self.shape)
        mat[:-1, :-2] = self.Du
        mat[:-1, -2]  = self.Da.reshape((self.Db.size, 1))
        mat[:-1, -1]  = self.Db.reshape((self.Da.size, 1))
        mat[-1, :]    = self.hxab

        return mat.tocsr()
