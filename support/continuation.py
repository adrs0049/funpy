#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np

from fun import Fun
from states.deflation_state import DeflationState
from states.parameter import Parameter
from colloc.chebOp import ChebOp


class Continuation:
    def __init__(self, op, pname, **kwargs):
        self.op = op
        self.pname = pname

    def execute(self, soln, start, stop, steps, debug=False, **kwargs):
        success = False
        pars = np.linspace(start, stop, steps + 1)
        for step, par in enumerate(pars):
            print(f'Step {step}: {self.pname} = {par:.4g}.')
            temp = DeflationState(u=soln, **{self.pname: par})
            soln, success, res = self.op.solve(temp, **kwargs)
            if debug: print(self.op.pars)

            if not success:
                break

        return soln, success
