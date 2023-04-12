#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
from enum import StrEnum

import numpy as np


class NewtonErrors(StrEnum):
    Success = 'Success',
    NonlinSolFail = 'NonlinSolFail',
    LinearSolFail = 'LinearSolFail',
    ContAngleFail = 'ContAngleFail',
    Undefined     = 'Undefined'


class NewtonBase:
    def __init__(self, system, LinOp, *args, **kwargs):
        """
        Arguments: Update.

        """
        # system is a class that provides an option to solve a linear problem
        self.system = system
        self.linop = LinOp
        self.linsys = None

        self.neval = 0
        self.normdx = 0.0
        self.debug = kwargs.pop('debug', False)
        self.verb = kwargs.pop('verb', False)
        self.status = NewtonErrors.Undefined

        self.normdxs = []

        # function type
        self.function_type = 'cheb'

    @property
    def dtype(self):
        if self.function_type == 'trig':
            return np.complex128
        return np.float64

    def setDisc(self, n):
        self.system.setDisc(n)
