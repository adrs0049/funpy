#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np

def minmaxCol(f, fp, xpts):
    # initialize
    pos = np.zeros((2, 1))
    vals = np.zeros((2, 1))

    # constant function
    if f.shape[0] == 1:
        vals[:, 0] = f(pos.flatten())

    # compute the turning points
    r = fp.roots()
    r = np.concatenate(([-1.0], r, [1.0]))
    v = f(r)

    # min
    idx = np.argmin(v)
    vals[0], pos[0] = v[idx], r[idx]

    # min with function values
    values = np.concatenate(([vals[0]], f.values))
    idx = np.argmin(values)
    min_value = values[idx]
    if min_value < vals[0]:
        vals[0] = min_value
        pos[0] = xpts[idx-1]

    # max
    idx = np.argmax(v)
    vals[1], pos[1] = v[idx], r[idx]

    # max with function values
    values = np.concatenate(([vals[1]], f.values))
    idx = np.argmax(values)
    max_value = values[idx]
    if max_value > vals[1]:
        vals[1] = max_value
        pos[1] = xpts[idx-1]

    return vals, pos
