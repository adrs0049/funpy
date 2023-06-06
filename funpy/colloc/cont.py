#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
from states.deflation_state import DeflationState


# simple class to do Natural parameter continuation
class Continuation:
    def __init__(self, op, pname):
        self.op = op
        self.pname = pname

    def __call__(self, guess, ps=None, no_steps=10, *args, **kwargs):
        # By default we use 10 continuation steps.
        ps = np.asarray(ps)
        if ps.size == 2:
            p0 = ps[0]
            pf = ps[1]
            dp = (pf - p0) / no_steps
            ps = np.arange(p0, pf + 0.5 * dp, dp)
        elif ps.size == 3:
            dp = ps[-1]
            ps = np.arange(p0, pf + 0.5 * dp, dp)

        for i in range(ps.size):
            print('Computing solution at {0:s} = {1:.4g}.'.format(self.pname, ps[i]))
            u0 = DeflationState(u=guess, **{self.pname: ps[i]})
            guess, success, res = self.op.solve(u0, verbose=True, adaptive=True, state=True)

            # HACK: This shouldn't be required anymore
            guess = guess.u

            if not success:
                print('Continuation failed!')
                break

        # Do not return the state object
        return guess
