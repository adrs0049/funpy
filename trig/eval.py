#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
from math import ceil

def horner_vec_cmplx(x, c):
    n = c.shape[0]
    nValsX = x.size

    if n == 1:
        return np.tile(c[0, :], (nValsX, 1))

    if len(x.shape) == 1:
        x = np.expand_dims(x, axis=1)

    z = np.exp(1j * np.pi * x)
    m = c.shape[1]
    if nValsX > 1:
        z = np.tile(z, (1, m))
        e = np.ones((nValsX, 1))
        q = np.tile(c[-1, :], (nValsX, 1))
        for j in range(n-2, 0, -1):
            q = e*c[j,:] + z * q

        if n & 1:
            q = np.exp(-1j*np.pi*(n-1)//2*x) * (e * c[0, :] + z * q)
        else:
            q = np.exp(-1j*np.pi*(n//2-1)*x) * q + np.cos(n//2 * np.pi * x) * c[0, :]
    else:
        q = c[-1, :]
        for j in range(n-2, 0, -1):
            q = z * q + c[j, :]

        if n & 1:
            q = np.exp(-1j*np.pi*(n-1)//2*x) * (z * q + c[0, :])
        else:
            q = np.exp(-1j*np.pi*(n//2-1)*x) * q + np.cos(n//2 * np.pi * x) * c[0, :]

    return q

def horner_scl_cmplx(x, c):
    n = c.shape[0]

    if n == 1:
        return np.ones(x.size) * c[0]

    z = np.exp(1j * np.pi * x)
    q = c[n-1]

    for j in range(n-2, 0, -1):
        q = c[j] + z * q

    if np.remainder(n, 2) == 1:
        q = np.exp(-1j * np.pi * (n-1)/2 * x) * (c[0] + z * q)
    else:
        q = np.exp(-1j * np.pi * (n/2-1) * x) * q + np.cos(n/2 * np.pi * x) * c[0]

    return q

def horner_scl_real(x, c):
    N = c.shape[0]

    # Get all negative indexed coefficients so that the constant is the first term
    n = ceil((N+1)/2)
    c = c[n-1::-1, :]
    a = np.real(c)
    b = np.imag(c)
    b.flags.writeable = True

    if np.remainder(N, 2) == 0:
        a[n-1] *= 0.5
        b[n-1] = 0

    if N == 1:
        return np.ones(x.size) * a

    u = np.cos(np.pi * x)
    v = np.sin(np.pi * x)
    co = a[n-1]
    si = b[n-1]

    for j in range(n-2, 0, -1):
        temp = a[j] + u * co + v * si
        si = b[j] + u * si - v * co
        co = temp

    return a[0] + 2 * (u * co + v * si)

def horner_vec_real(x, c):
    nValsX = x.shape[0]
    N = c.shape[0]
    numCols = c.shape[1]

    # Get all negative indexed coefficients so that the constant term is the first term
    n = ceil((N+1)/2)
    c = c[n-1::-1, :]
    a = np.real(c)
    b = np.imag(c)
    e = np.ones((nValsX, 1))

    if np.remainder(N, 2) == 0:
        a[n-1, :] *= 0.5
        b[n-1, :] = 0

    if N == 1:
        return np.matmul(e, a)

    u = np.tile(np.cos(np.pi * x), (1, numCols))
    v = np.tile(np.sin(np.pi * x), (1, numCols))
    co = np.matmul(e, a[None, n-1, :])
    si = np.matmul(e, b[None, n-1, :])

    for j in range(n-2, 0, -1):
        temp = np.matmul(e, a[None, j, :]) + u * co + v * si
        si = np.matmul(e, b[None, j, :]) + u * si - v * co
        co = temp

    return np.matmul(e, a[None, 0, :]) + 2 * (u * co + v * si)

def horner(x, c, isReal):
    scalarValued = c.shape[1] == 1

    if scalarValued and np.all(isReal):
        return horner_scl_real(x, c)
    elif ~scalarValued and np.all(isReal):
        return horner_vec_real(x, c)
    elif scalarValued and ~np.all(isReal):
        return horner_scl_cmplx(x, c)
    else:
        out = horner_vec_cmplx(x, c)
        out[:, isReal] = np.real(out[:, isReal])
        return out
