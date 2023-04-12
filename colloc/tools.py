#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np


def findNextPowerOf2(n):
    if n <= 1:
        return 2

    n = n - 1
    while n & n - 1:
        n = n & n - 1

    return n << 1


def remove_bc(iar, n_eqn, bw=2):
    n = iar.size // n_eqn
    m = n_eqn * (n - bw)

    out = np.empty(m, dtype=iar.dtype)

    for i in range(n_eqn):
        for j in range(n - bw):
            out[i * (n - bw) + j] = iar[i * n + j + bw]

    return out.reshape((n - bw, n_eqn), order='F')


def replace_bc(iar, n_eqn, bw=2, reshape=False):
    n = iar.size // n_eqn

    out = np.zeros_like(iar)

    for i in range(n_eqn):
        for j in range(n - bw):
            out[i * n + j] = iar[i * n + j + bw]

    if reshape:
        return out.reshape((n, n_eqn), order='F')
    else:
        return out


def get_bc(iar, n_eqn, bw=2):
    n = iar.size // n_eqn
    out = np.zeros(n_eqn * bw, dtype=float)

    for i in range(n_eqn):
        for j in range(bw):
            out[i * bw + j] = iar[i * n + j]

    return out
