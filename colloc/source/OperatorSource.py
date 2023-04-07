#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np

from ac.gen import CodeGeneratorBackend

from sympy import sympify
from sympy.matrices import zeros, ones

from colloc.source.OperatorSourceBlock import OperatorSourceBlock


class OperatorSource:
    def __init__(self, shape, *args, **kwargs):
        self.name = kwargs.pop('name', 'Unknown')
        self.func_names = kwargs.pop('func_names', [])
        self.cfunc_names = kwargs.pop('cfunc_names', [])
        self.pfuns = kwargs.pop('pfun', [])
        self.domain = kwargs.pop('domain', np.asarray([-1, 1]))
        self.pars = kwargs.pop('pars', [])
        self.ftype = kwargs.pop('ftype', 'cheb')

        self.ops = np.empty(shape, dtype=object)
        for i, j in np.ndindex(self.ops.shape):
            self.ops[i, j] = OperatorSourceBlock(i, j, op=self)

    @property
    def neqn(self):
        return len(self.func_names)

    @property
    def shape(self):
        return self.ops.shape

    @property
    def expr(self):
        n, m = self.shape
        expr = zeros(n, m)
        for i, j in np.ndindex(self.shape):
            expr[i, j] = self[i, j].expr

        return expr

    @property
    def action(self):
        """
        Returns the expression of the action of the linear operator
        represented by this sympy code.

        If the operator is a block matrix like this it returns this action:

            / F_11 F_12 \  / u \
            |           |  |   |
            \ F_21 F_22 /  \ v /

        """
        matrix = self.expr
        expr = zeros(self.shape[0], 1)
        temp_vector = ones(rows=1, cols=self.shape[1])
        for i in range(self.shape[0]):
            expr[i] = matrix.row(i).dot(temp_vector)

        return expr

    @property
    def biaction(self):
        """
        Returns the expression of the biaction of the linear operator
        represented by this sympy code.

        If the operator is a block matrix like this it returns this biaction:

                    / F_11 F_12 \  / u2 \
         < u1 v1 >  |           |  |    |
                    \ F_21 F_22 /  \ v2 /

        """
        matrix = self.expr
        expr = sympify(0)
        for i, j in np.ndindex(matrix.shape):
            expr += matrix[i, j]

        return expr

    @property
    def transpose(self):
        """ Returns the transpose of the operator """
        matrix = self.expr.transpose()
        for i, j in np.ndindex(matrix.shape):
            if i == j: continue
            matrix[i, j] = matrix[i, j].replace(self.pfuns[i], self.pfuns[j])

        return matrix

    def finish(self):
        for i, j in np.ndindex(self.shape):
            self.ops[i, j].finish()

    def __getitem__(self, idx):
        try:
            src = self.ops[idx]
        except Exception as e:
            raise e
        return src

    def __repr__(self):
        rstr = type(self).__name__ + ' \"{0:s}\"'.format(self.name) + ':\n'
        for i, j in np.ndindex(self.shape):
            rstr += repr(self[i, j])
        return rstr

    def __str__(self):
        return self.__repr__()
