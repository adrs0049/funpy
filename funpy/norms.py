#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np

from .fun import norm2
from .vectorspaces import innerw


def nnorm(u, scale=None, weighted=True, p=2):
    if scale is None:
        return u.norm(p=p, norm_function=norm2, weighted=weighted)
    else:
        return (u * scale).norm(p=p, norm_function=norm2, weighted=weighted)


def sprod(v1, v2, weighted=False, scale=None):
    if weighted:
        if scale is not None:
            rval = innerw(v1 * scale, v2 * scale)
        else:
            rval = innerw(v1, v2)

        if rval.size > 1:
            rval = np.sum(rval)
    else:
        if scale is not None:
            rval = np.inner(v1 * scale, v2 * scale)
        else:
            rval = np.inner(v1, v2)

        # extract result
        if rval.size > 1:
            rval = np.sum(np.diagonal(rval))

    # TODO: checkme!
    return np.real(rval)


def rescale(x, xa, xthresh):
    return 1. / 0.5 * (x + xa)
