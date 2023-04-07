#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as LAS
from fun import Fun
from colloc.ultraS.matrices import convertmat


def chebpoly(n, domain=[-1, 1], kind=1):
    assert kind == 1 or kind == 2, ''
    if not isinstance(n, list):
        n = [n]

    N = np.max(n) + 1
    c = sps.eye(N).tolil()
    c = c[:, n]

    if kind == 2:
        S = convertmat(N, 0, 0, format='csc')
        c = LAS.spsolve(S, c.tocsc())

    c = np.asfortranarray(c.todense())
    return Fun.from_coeffs(c, n_funs=len(n), domain=domain)
