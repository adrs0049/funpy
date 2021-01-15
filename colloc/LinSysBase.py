#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
from ac.support import Namespace

from fun import Fun


class LinSysBase:
    """
        TODO: write me!
    """
    def __init__(self, diffOrder: int, *args, **kwargs):
        self.ns = Namespace()
        self.n_disc = kwargs.get('n_disc', 15)
        self.par = kwargs.pop('par', False)

        # store this
        self.diffOrder = diffOrder
        self.projOrder = kwargs.pop('projOrder', diffOrder)

        # Operator constraints such as boundary conditions
        self.constraints = []

        # Fredholm operator Constraints -> i.e. mass constraints!
        self.conditions = []

    @property
    def dshape(self):
        return (self.n_disc, self.linOp.shape[1])

    @property
    def pars(self):
        """ Prints the current parameters in the namespace! """
        return self.ns['print_pars']()

    @property
    def numConstraints(self):
        return len(self.constraints)

    def setParametersPartial(self, pars):
        # update the namespace
        try:
            for p_name, p_value in pars.update_items():
                self.ns[p_name] = p_value
        except AttributeError:
            pass

    def setParameters(self, pars):
        # update the namespace
        try:
            for p_name, p_value in pars.items():
                self.ns[p_name] = p_value
        except AttributeError:
            pass

    def getDimAdjust(self):
        """ Returns an array of the same size as data, which tells the discretization how to increase / decrease dimension
            ensure that highest order derivative is discretized at the same order at the same dimension
        """
        adjust = max(0, self.getDiffOrder())
        return np.tile(adjust, self.linOp.shape[1])

    def getDiffOrder(self) -> int:
        return 0

    def getOutputSpace(self) -> int:
        return self.diffOrder

    def getProjOrder(self):
        # TODO not ideal!
        return self.projOrder * np.ones(self.linOp.shape[0])
