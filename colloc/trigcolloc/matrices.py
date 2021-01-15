#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
from functools import lru_cache
import scipy.sparse as sps
import scipy.linalg as LA
from scipy.fft import ifft, fft
from scipy.sparse import spdiags, eye, csr_matrix, lil_matrix, issparse
from sparse.csr import delete_rows_csr, eliminate_zeros_csr

from colloc.ultraS.transform import sptoeplitz, spconvert, spconvert_inv, sphankel
from cheb.chebpts import chebpts_type2, barymat, quadwts
from cheb.simplify import standardChop

CACHE_SIZE = 25

#@lru_cache(maxsize=CACHE_SIZE)
def diffmat(n, m=1, format=None):
    """ Differentiation matrix for trigonometric spectral method

    The operator maps the function values of a trigonometric function on an equidistant points in
    [-1, 1) to the values of the derivative of this function at the same points.
    """
    if n == 0:
        return NotImplemented
    elif n == 1:
        return np.asarray(0)

    # Grid point spacing
    h = 2. * np.pi / n

    if m == 0:
        return eye(n, format=format)
    elif m == 1:
        v = np.arange(1, n)
        if np.remainder(n, 2):  # n is odd
            csc = 0.5 / np.sin(0.5 * v * h)
            column = np.hstack((0, csc))
        else:  # n is even
            cot = 0.5 / np.tan(0.5 * v * h)
            column = np.hstack((0, cot))

        # flip sign of every second element
        column[1::2] *= -1.0
        row = column[np.roll(np.arange(n)[::-1], 1)]
        D = LA.toeplitz(column, row)

    elif m == 2:
        v = np.arange(1, n)
        if np.remainder(n, 2):  # n is odd
            csc_cot = (0.5/np.sin(0.5 * v * h)) * (1./np.tan(0.5 * v * h))
            column = np.hstack((np.pi**2 / 3. / h**2 - 1./12, csc_cot))
        else:  # n is even
            csc = 0.5 / np.sin(0.5 * v * h)**2
            column = np.hstack((np.pi**2 / 3. / h**2 + 1./6, csc))

        column[::2] *= -1.0
        D = LA.toeplitz(column)

    elif m == 3:
        v = np.arange(1, n)
        if np.remainder(n, 2):  # n odd
            csc = 1. / np.sin(0.5 * v * h)
            cot = 1. / np.tan(0.5 * v * h)
            column = np.hstack((0, 3./8 * csc * cot**2 + 3./8 * csc**3 - np.pi**2 / 2 / h**2 * csc))
        else:
            csc_cot = (1. / np.sin(0.5 * v * h)**2) * (1. / np.tan(0.5 * h * v))
            column = np.hstack((0, 3./4 * csc_cot - np.pi**2 / 2 / h**2 / np.tan(0.5 * h * v)))

        column[::2] *= -1.0
        row = column[np.roll(np.arange(n)[::-1], 1)]
        D = LA.toeplitz(column, row)

    elif m == 4:
        v = np.arange(1, n)
        csc = 1. / np.sin(0.5 * h * v)
        cot = 1. / np.tan(0.5 * h * v)
        if np.remainder(n, 2):  # n is odd
            column = np.hstack(())
        else:   # n even
            column = np.hstack(())

        column[::2] *= -1.0
        D = LA.toeplitz(column)

    else:
        if np.remainder(n, 2):  # n odd
            column = np.hstack(())
        else:
            column = np.hstack(())

        D = np.real(ifft(column * fft(eye(n))))

    # scale from [-pi, pi) to [-1, 1)
    D *= (np.pi**m)
    return D

def multmat(k, f, lam, chop=False, eps=1.48e-8):
    """ Multiplication matrices for ultraS

        This function forms the k x k multiplication matrix representing the multiplication of F in the C^(lambda) basis.
    """
    pass

def intmat(k, format=None):
    """ Integration matrices for ultraS

        This function forms the k x k integration matrix representing the integration of a function
        F expressed using in a C^(0) basis.

        The block structure of this matrix is as follows:

            /   CCW   \
            |    0    |
            \    0    /

        i.e. a matrix having only a first non-zero row, with the first row being filled by the
        Curtis-Clenshaw weights for integration.
    """
    pass

def evalfun(n, location):
    """ The evaluation functional """
    # find the collocation points and create an empty functional
    x, _, v, _ = chebpts_type2(n)
    E = np.zeros((1, n))
    E[0, :] = barymat(location, x, v)
    return E

""" Some helper functions that allow easy rectangularization of matrices """
def delete_rows(matrix, rows_to_delete):
    if issparse(matrix):
        return delete_rows_csr(matrix, rows_to_delete)
    else:
        return np.delete(matrix, rows_to_delete, axis=0)

def blockmat(matrix, **kwargs):
    # need to check the blocks to see whether they are sparse!
    isVector = False
    for block in matrix.flat:
        if block.shape[1] == 1:
            isVector = True

    # Depending on what type of matrices we find call the appropriate block
    # matrix constructor
    if isVector:
        # if we have column vectors we simply need to append those
        return np.vstack(matrix.flatten())
    else:
        return sps.bmat(matrix, **kwargs)
