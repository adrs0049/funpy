#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
from colloc.chebOp import chebOp

# simple class to do Natural parameter continuation
class Continuation:
    def __init__(self, op, pname):
        self.op = op
        self.pname = pname

    def __call__(self, guess, ps=None, *args, **kwargs):
        # By default we use 10 continuation steps.
        ps = np.asarray(ps)
        if ps.size == 2:
            p0 = ps[0]
            pf = ps[1]
            dp = (pf - p0) / 10
            ps = np.arange(p0, pf + dp, dp)
        elif ps.size == 3:
            dp = ps[-1]
            ps = np.arange(p0, pf + dp, dp)

        for i in range(ps.size):
            guess, success, res = self.op.solve(guess, cpar=self.pname, state=True, method='lgmres')

            if not success:
                print('Continuation failed! Solver reported |res| = %.4g.' % res)
                break

            # Update the continuation parameter
            if i < ps.size-1: guess.a = ps[i+1]

        # Do not return the state object
        return guess.u
