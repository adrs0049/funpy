#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
from functools import lru_cache
import scipy.sparse as sps
from sparse.csr import delete_rows_csr, eliminate_zeros_csr
from scipy.sparse import spdiags, eye, csr_matrix, lil_matrix, issparse

from funpy.colloc.ultraS.transform import sptoeplitz, spconvert, spconvert_inv, sphankel
from funpy.trig.trigpts import quadwts

CACHE_SIZE = 25

# @lru_cache(maxsize=CACHE_SIZE)
def diffmat(n, m=1, format=None, flag=False):
    """ Differentiation matrices for trigspec spectral method

    This matrix maps N Fourier coefficients and returns N coefficients that represent the m-th
    derivative.
    """
    if m > 0:
        if np.remainder(n, 2):  # N odd
            D = (1j)**m * spdiags(np.arange(-(n-1)/2, n/2, 1), 0, n, n, format=format)**m
        else:   # N even
            if np.remainder(m, 2):  # m odd
                D = (1j)**m * spdiags(np.hstack((0, np.arange(-n/2+1, n/2, 1))), 0, n, n, format=format)**m
                if flag:
                    # set the (0, 0) entry to (-1j * N/2)^m, instead of zero
                    D[0, 0] = (-1j * n / 2)**m
            else:  # m even
                D = (1j)**m * spdiags(np.arange(-n/2, n/2, 1), 0, n, n, format=format)**m

        return D
    elif m == 0:
        return eye(n, format=format)
    else:
        return NotImplemented

def multmat(k, f, format=None):
    """ Multiplication matrices for trigspec

        Forms the kxk matrix that represents function multiplication of F in a Fourier basis.
    """
    # Can only handle with non-array functions!
    assert f.shape[1] == 1, 'multmat can only deal with scalar trigtechs!'

    n = f.shape[0]
    c = f.coeffs.flatten()

    # Multiplication using a scalar is simple!
    if n == 1:
        # the function f is a scalar
        return np.asscalar(c) * eye(k)

    # Deal with the even case
    if np.remainder(n, 2) == 0:
        c = np.hstack((0.5 * c[0], c[1:], 0.5 * c[0]))

    # update the value of n
    n = c.shape[0]

    # Position of the constant term
    Na = np.floor(n/2).astype(int)

    # Pad with zeros
    if Na < k:
        col = np.hstack((c[Na:],    np.zeros(k-Na-1)))
        row = np.hstack((c[Na::-1], np.zeros(k-Na-1)))
    else:  # truncate FIXME!
        col = c[Na:]
        row = c[Na::-1]

    return sptoeplitz(col, row, format=format)

def intmat(k, format=None):
    """ Integration matrices for ultraS

        This function forms the k x k integration matrix representing the integration of a function
        F expressed using in a C^(0) basis.

        The block structure of this matrix is as follows:

            /   QuadWts   \
            |      0      |
            \      0      /

        i.e. a matrix having only a first non-zero row, with the first row being filled by the
        quadrature rule for trapezoidal integration. In this case only the constant coefficient
        should contribute to the integral value! But we need to find its location.
    """
    # Index of the constant coefficients
    if np.remainder(k, 2):  # k is odd
        ind = (k+1)//2 - 1
    else:
        ind = k//2

    # Only the constant coefficient contributes to the integral's value
    w = np.zeros(k)
    w[ind] = 2.

    # Create the matrix for the operator
    I = lil_matrix((k, k))
    I[ind, :] = w
    return I.asformat(format)

def gen_m0(i, Rscl):
    return 1j * np.piecewise(i, [i == 0, (i < 0) | (i > 0)],
                             [lambda k: 0, lambda k: (1.0 - np.cos(Rscl * k)) / (Rscl * k)])

def gen_m1(i, Rscl):
    return np.piecewise(i, [i == 0, (i < 0) | (i > 0)],
                        [lambda k: 0, lambda k: (1.0 - np.cos(Rscl * k)) / (Rscl)])

def aggmat(n, m=0, rescaleFactor=1, format=None):
    """ Generates the matrix for the aggregation operator.
        Here the aggregation operator is assumed to use the non-smooth uniform integration kernel.

        - rescaleFactor required here since the non-local operator always interacts with the domain
        size, and thus the matrices coefficients depend in a non-linear way on the rescale factor.

        m -> order of the derivative operator in front of the non-local operator.
    """
    assert m <= 1 and m >= 0, 'Aggregation matrix only supports m = 0, 1; requested m = %d.' % m
    Rscl = rescaleFactor * np.pi

    # Silence warnings / exceptions that we will encounter next! But don't worry we will fix it!
    if np.remainder(n, 2):  # n is odd
        k = np.arange(-(n-1)/2, n/2, 1)
    else:  # n is even
        k = np.arange(-n/2, n/2, 1)

    if m == 0:
        even_odd_fix = gen_m0(k, Rscl)
    elif m == 1:
        # Same as the above but we multiply by ik
        even_odd_fix = gen_m1(k, Rscl)
    else:  # Shouldn't happen!
        return NotImplemented

    return spdiags(even_odd_fix, 0, n, n, format=format)

def evalfun(n, location):
    """ The evaluation functional """
    return NotImplemented

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
