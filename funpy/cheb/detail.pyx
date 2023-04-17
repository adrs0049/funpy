# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
#cython: language_level=3
#cython: annotate=True
#cython: infer_types=True

import numpy as np
import scipy.linalg as LA
import functools

cimport numpy as np
cimport cython
cimport detail

# Import this directly -> it's much faster!
from scipy.fft._pocketfft.pypocketfft import dct
from scipy.fft import ifft, fft
from scipy.sparse import spdiags, eye, triu

from libc.math cimport sqrt, abs, log, log10, fmax, round

# Local imports
from .pts import chebpts_type2_compute

# TYPE DEFINITIONS
cb_t = np.double
ctypedef np.double_t DTYPE_t

# Helper functions
cdef inline DTYPE_t r(DTYPE_t e1, DTYPE_t tol):
    """ """
    return 3.0 * (1.0 - log(e1)/log(tol))


cdef inline Py_ssize_t max_idx(Py_ssize_t a, Py_ssize_t b):
    return b if (a < b) else a


cdef inline Py_ssize_t min_idx(Py_ssize_t a, Py_ssize_t b):
    return a if (a < b) else b


def chebptsAB(n, ab):
    # TODO: eventually generalize this again!
    x = chebpts_type2_compute(n)
    return 0.5 * (ab[1] * (x + 1) + ab[0] * (1 - x))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef polyfit(const double[:, :] sampled, int workers = 1):
    """
    Compute Chebyshev coefficients for values located on Chebyshev points.
    sampled: array; first dimension is number of Chebyshev points
    """
    N = sampled.shape[0]
    M = sampled.shape[1]
    if N == 1:
        return np.copy(sampled, order='F')

    c = np.empty((N , M), dtype=cb_t, order='F')
    cdef double[:, :] cv = c
    cdef Py_ssize_t i, j

    # Flip and re-scale in one go
    for j in range(M):
        for i in range(N):
            cv[i, j] = sampled[N - i - 1, j] / (N - 1)

    # Compute the DCT-I
    dct(c, 1, (0, ), 0, c, workers)

    for j in range(M):
        cv[0, j] /= 2.0
        cv[N-1, j] /= 2.0

    return c


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef polyval(const double[:, :] chebcoeff, int workers = 1):
    """ compute the interpolation values at chebyshev points. """
    N = chebcoeff.shape[0]
    M = chebcoeff.shape[1]
    if N == 1:
        return np.copy(chebcoeff, order='F')

    v = np.empty((N , M), dtype=cb_t, order='F')
    cdef double[:, :] vv = v
    cdef Py_ssize_t i, j

    # Simply copy the first and last row
    for j in range(M):
        vv[0, j] = chebcoeff[0, j]
        vv[N-1, j] = chebcoeff[N-1, j]

    # Scale all the remaining rows
    for j in range(M):
        for i in range(1, N - 1):
            vv[i, j] = chebcoeff[i, j] / 2.0

    # Apply the DCT-I in v
    dct(v, 1, (0, ), 0, v, workers)

    # Finally flip everything around!
    return np.flipud(v)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef prolong(double[::1, :] array, int Nout):
    # If Nout < length(self) -> compressed by chopping
    # If Nout > length(self) -> coefficients are padded by zero
    cdef Py_ssize_t Nin = array.shape[0]
    cdef Py_ssize_t M = array.shape[1]
    cdef Py_ssize_t Ndiff = Nout - Nin
    cdef Py_ssize_t i, j

    if Ndiff == 0: # Do nothing
        return array

    cdef double[::1, :] av
    if Ndiff > 0:  # pad with zeros
        a = np.empty((Nout, M), dtype=cb_t, order='F')
        av = a.view()

        for j in range(M):
            for i in range(Nin):
                av[i, j] = array[i, j]

        for j in range(M):
            for i in range(Nin, Nout):
                av[i, j] = 0.0

        return a

    else:  # Ndiff < 0
        Nout = max_idx(Nout, 0)
        return array[:Nout, :]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef simplify_coeffs(double[::1, :] coeffs, eps=1.48e-8):
    cdef Py_ssize_t nold = coeffs.shape[0]
    cdef Py_ssize_t N = max_idx(17, <Py_ssize_t>round(nold * 1.25 + 5))
    cdef Py_ssize_t m = coeffs.shape[1]
    cdef Py_ssize_t cutoff, k

    # Elongate to required minima
    coeffs = prolong(coeffs, N)
    coeffs = polyfit(polyval(coeffs))

    # get tolerances
    tol = np.max(eps) * np.ones(m)

    # loop through columns to compute cutoff
    cutoff = 0
    for k in range(m):
        chop = standardChop(coeffs[:, k], tol[k])
        cutoff = max_idx(cutoff, chop)

    # take the minimum cutoff.
    cutoff = min_idx(cutoff, nold)

    # chop coefficients: +1 here since cutoff is the last index to keep!
    return np.asarray(coeffs[:cutoff+1, :], order='F')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef happiness_check(double[::1, :] coeffs, double [::1] tol, Py_ssize_t minimum=17):
    cdef Py_ssize_t n = coeffs.shape[0]
    cdef Py_ssize_t m = coeffs.shape[1]
    cdef Py_ssize_t cutoff, k, rcutoff = 0
    cdef int ishappy

    # Elongate to required minima for standardChop
    # Otherwise the happiness check returns silly results.
    if n < minimum:
        coeffs = prolong(coeffs, minimum)
        coeffs = polyfit(polyval(coeffs))

    for k in range(m):
        cutoff = standardChop(coeffs[:, k], tol[k])
        ishappy = (cutoff < n)
        rcutoff = max(rcutoff, cutoff)
        if not ishappy:
            break

    return ishappy, rcutoff


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef standardChop(double[::1] coeffs, double tol, Py_ssize_t minimum=17):
    """
        coeffs: Column vector of real or complex numbers which are either
        Chebyshev or Fourier coefficients.

        TOL: a number between (0, 1) representing a target relative accuracy.

        Output: Cutoff: a positive integer
            If cutoff == length(coeffs), then we are not happy

            if cutoff < length(cutoff), we are happy and cutoff represents the
            last index of coeffs that should be retained
    """
    # Define variables
    cdef Py_ssize_t cutoff, plateauPoint, d, j, j2, j3
    cdef double e1, e2, ptol

    if tol >= 1:
        cutoff = 0
        return cutoff

    cdef Py_ssize_t n = coeffs.shape[0]
    cutoff = n

    # we only chop when the length is longer than 17
    if n < minimum:
        return cutoff

    # Step 1: Convert coeffs to a new monotonically nonincreasing vector
    # envelope normalized to begin with the value 1.
    cdef np.ndarray[DTYPE_t, ndim=1] envelop = np.empty(n, dtype=cb_t)
    cdef np.ndarray[DTYPE_t, ndim=1] cc

    cdef DTYPE_t abs_value
    abs_value = abs(coeffs[n-1])
    for j in range(n): envelop[j] = abs_value

    for j in range(n-2, -1, -1):
        envelop[j] = fmax(abs(coeffs[j]), envelop[j+1])

    if envelop[0] == 0:
        cutoff = 0
        return cutoff

    # normalize - envelop function
    for j in range(n):
        envelop[j] /= envelop[0]

    # Step 2: Scan the envelope for a value PLAEAUPOINT.
    for j in range(1, n):
        j2 = <Py_ssize_t>round(1.25 * j + 5)
        if j2 >= n: # there is no plateau: exit
            return cutoff

        e1 = envelop[j]
        e2 = envelop[j2]
        if e1 == 0 or e2/e1 > r(e1, tol):
            plateauPoint = j - 1
            break

    # Step 3: Fix cutoff at a point where envelope, plus a linear function
    # included to bias the result towards the left and is minimal
    if envelop[plateauPoint] == 0:
        cutoff = plateauPoint
    else:
        j3 = 0
        ptol = tol**(7./6)
        for j in range(n):
            if envelop[j] >= ptol: j3 += 1

        if j3 < j2:
            j2 = j3 + 1
            envelop[j2-1] = ptol

        cc = np.log10(envelop[:j2])
        cc += np.linspace(0, (-1.0/3) * log10(tol), j2)
        d = np.argmin(cc)
        cutoff = max_idx(d - 1, 0)

    return cutoff


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef standardChopCmplx(complex[::1] coeffs, double tol, Py_ssize_t minimum=17):
    """
        coeffs: Column vector of real or complex numbers which are either
        Chebyshev or Fourier coefficients.

        TOL: a number between (0, 1) representing a target relative accuracy.

        Output: Cutoff: a positive integer
            If cutoff == length(coeffs), then we are not happy

            if cutoff < length(cutoff), we are happy and cutoff represents the
            last index of coeffs that should be retained

        - TODO: this function really only replaces the abs function.
    """
    # Define variables
    cdef Py_ssize_t cutoff, plateauPoint, d, j, j2, j3
    cdef double e1, e2, ptol

    if tol >= 1:
        cutoff = 0
        return cutoff

    cdef Py_ssize_t n = coeffs.shape[0]
    cutoff = n

    # we only chop when the length is longer than 17
    if n < minimum:
        return cutoff

    # Step 1: Convert coeffs to a new monotonically nonincreasing vector
    # envelope normalized to begin with the value 1.
    cdef np.ndarray[DTYPE_t, ndim=1] envelop = np.empty(n, dtype=cb_t)
    cdef np.ndarray[DTYPE_t, ndim=1] cc

    cdef DTYPE_t abs_value
    abs_value = sqrt(coeffs[n-1].real * coeffs[n-1].real + coeffs[n-1].imag * coeffs[n-1].imag)
    for j in range(n): envelop[j] = abs_value

    for j in range(n-2, -1, -1):
        envelop[j] = fmax(sqrt(coeffs[j].real * coeffs[j].real + coeffs[j].imag *
                               coeffs[j].imag), envelop[j+1])

    if envelop[0] == 0:
        cutoff = 0
        return cutoff

    # normalize - envelop function
    for j in range(n):
        envelop[j] /= envelop[0]

    # Step 2: Scan the envelope for a value PLAEAUPOINT.
    for j in range(1, n):
        j2 = <Py_ssize_t>round(1.25 * j + 5)
        if j2 >= n: # there is no plateau: exit
            return cutoff

        e1 = envelop[j]
        e2 = envelop[j2]
        if e1 == 0 or e2/e1 > r(e1, tol):
            plateauPoint = j - 1
            break

    # Step 3: Fix cutoff at a point where envelope, plus a linear function
    # included to bias the result towards the left and is minimal
    if envelop[plateauPoint] == 0:
        cutoff = plateauPoint
    else:
        j3 = 0
        ptol = tol**(7./6)
        for j in range(n):
            if envelop[j] >= ptol: j3 += 1

        if j3 < j2:
            j2 = j3 + 1
            envelop[j2-1] = ptol

        cc = np.log10(envelop[:j2])
        cc += np.linspace(0, (-1.0/3) * log10(tol), j2)
        d = np.argmin(cc)
        cutoff = max_idx(d - 1, 0)

    return cutoff

""" Clenshaw for scalar equations """
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef clenshaw_scalar(np.ndarray[DTYPE_t, ndim=1] x, np.ndarray[DTYPE_t, ndim=1] c):
    cdef Py_ssize_t N = x.shape[0]
    cdef Py_ssize_t n = c.shape[0] - 1
    cdef Py_ssize_t i, j
    cdef np.ndarray[DTYPE_t, ndim=1] bk1 = np.zeros(N)
    cdef np.ndarray[DTYPE_t, ndim=1] bk2 = np.zeros(N)
    cdef DTYPE_t tmp

    x = 2*x
    for i in range(n, 1, -2):
        for j in range(N):
            bk2[j] = c[i]   + x[j] * bk1[j] - bk2[j]
            bk1[j] = c[i-1] + x[j] * bk2[j] - bk1[j]

    if n & 1:
        for j in range(N):
            tmp = bk1[j]
            bk1[j] = c[1] + x[j] * bk1[j] - bk2[j]
            bk2[j] = tmp

    # make sure this returns a vector that is inline with what clenshaw_vector would return
    # write the answer into the buffer bk1
    for j in range(N):
        bk1[j] = c[0] + 0.5 * x[j] * bk1[j] - bk2[j]

    return np.expand_dims(bk1, axis=1)

""" Clenshaw for vector equations """
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef clenshaw_vector(np.ndarray[DTYPE_t, ndim=1] x, np.ndarray[DTYPE_t, ndim=2] c):
    cdef Py_ssize_t N = x.size
    cdef Py_ssize_t n = c.shape[0] - 1
    cdef Py_ssize_t m = c.shape[1]
    cdef Py_ssize_t i, j, k

    cdef np.ndarray[DTYPE_t, ndim=2] bk1 = np.zeros((N, m), dtype=cb_t)
    cdef np.ndarray[DTYPE_t, ndim=2] bk2 = np.zeros((N, m), dtype=cb_t)
    cdef DTYPE_t tmp

    x = 2*x
    for i in range(n, 1, -2):
        for j in range(N):
            for k in range(m):
                bk2[j, k] = c[i, k]   + x[j] * bk1[j, k] - bk2[j, k]
                bk1[j, k] = c[i-1, k] + x[j] * bk2[j, k] - bk1[j, k]

    if n & 1:
        for j in range(N):
            for k in range(m):
                tmp = bk1[j, k]
                bk1[j, k] = c[1, k] + x[j] * bk1[j, k] - bk2[j, k]
                bk2[j, k] = tmp

    for j in range(N):
        for k in range(m):
            bk1[j, k] = c[0, k] + 0.5 * x[j] * bk1[j, k] - bk2[j, k]

    return bk1

cpdef clenshaw(x, coeffs):
    if len(coeffs.shape) == 1:
        return clenshaw_scalar(np.atleast_1d(x), coeffs)
    elif coeffs.shape[1] == 1:
        return clenshaw_scalar(np.atleast_1d(x), np.reshape(coeffs, coeffs.shape[0]))
    else:
        return clenshaw_vector(np.atleast_1d(x), coeffs)

""" Computing TLeft and TRight is expensive -> so we will only do this once """
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@functools.lru_cache(maxsize=10)
def fft_weights(double splitPoint, Py_ssize_t n=513):
    # Create coefficients for TLeft using FFT
    cdef Py_ssize_t i, j
    cdef np.ndarray[DTYPE_t, ndim=1] xLeft  = chebptsAB(513, [-1, splitPoint])
    cdef np.ndarray[DTYPE_t, ndim=1] xRight = chebptsAB(513, [splitPoint, 1])
    cdef np.ndarray[DTYPE_t, ndim=2] TLeft  = np.ones((513, 513), dtype=cb_t)
    cdef np.ndarray[DTYPE_t, ndim=2] TRight = np.ones((513, 513), dtype=cb_t)

    # Create memory views to the various arrays
    cdef double[:] xlv = xLeft
    cdef double[:] xrv = xRight
    cdef double[:, :] lv = TLeft
    cdef double[:, :] rv = TRight

    for j in range(513):
        lv[j, 1] = xlv[j]

    for i in range(2, 513):
        for j in range(513):
            lv[j, i] = 2 * xlv[j] * lv[j, i-1] - lv[j, i-2]

    for j in range(513):
        rv[j, 1] = xrv[j]

    for i in range(2, 513):
        for j in range(513):
            rv[j, i] = 2 * xrv[j] * rv[j, i-1] - rv[j, i-2]

    # This is less expensive than the above -> and most time is spent in the fft!
    TLeft = np.vstack((TLeft[513:0:-1], TLeft[:512, :]))
    TLeft = np.real(fft(TLeft, axis=0) / 512)
    TLeft = np.triu(np.vstack((0.5 * TLeft[0, :],
                               TLeft[1:512, :],
                               0.5 * TLeft[512, :])))

    TRight = np.vstack((TRight[513:0:-1], TRight[:512, :]))
    TRight = np.real(fft(TRight, axis=0) / 512)
    TRight = np.triu(np.vstack((0.5 * TRight[0, :],
                                TRight[1:512, :],
                                0.5 * TRight[512, :])))

    return TLeft, TRight


def roots_main(c, htol, splitPoint, eps=1.48e-8,
               all_roots=False, prune_roots=False):
    """ Computes the roots of the polynomial given by the coefficients c on
        the unit interval
    """
    maxEigSize = 50

    # simplify the coefficients
    tailMmax = 5 * eps * LA.norm(c, ord=1)

    # find the final coefficient about tailMax:
    n = c.size
    if n > 1:
        n = np.max(np.where(np.abs(c) > tailMmax)[0]) + 1

        # Truncate the coefficients (rather than alias):
        if n is not None and n > 1 and n < c.size:
            c = c[:n]

    # print('ct:', c.size)
    # print('n:', n)

    # Trivial return type
    r = []

    # Trivial cases
    if n == 0:
        r = [0]

    elif n == 1:
        if c[0] == 0:
            r = [0]
        else:
            r = []

    elif n == 2:
        r = [-c[0] / c[1]]

        if all_roots:
            if np.abs(np.imag(r)) > htol or (r < -(1 + htol)) or (r > (1 + htol)):
                r = []
            else:
                r = [max(min(np.real(r), 1), -1)]

    elif n <= maxEigSize:
        # Adjust the coefficients for the colleague matrix:
        oh = 0.5 * np.ones(n-2)
        A = spdiags(np.vstack((oh, oh)), (1, -1), n-1, n-1).todense()
        A[-2, -1] = 1

        # Setup GEP for extra stability
        B = eye(n-1).todense()
        c /= LA.norm(c, np.inf)
        B[0, 0] = c[-1]
        c = -0.5 * c[:-1]
        c[-2] = c[-2] + 0.5 * B[0, 0]
        A[:, 0] = np.expand_dims(np.flipud(c), axis=1)

        # compute the roots: Is there a sparse variant of this?
        r = LA.eigvals(A, b=B)

        # clean-up the roots
        if not all_roots:
            # remove dangling imag parts:
            mask = np.abs(np.imag(r)) < htol
            r = np.real(r[mask])
            r = np.sort(r[np.abs(r) <= 1 + htol])
            if r.size > 0:
                r[0] = max(r[0], -1)
                r[-1] = min(r[-1], 1)

        elif prune_roots:
            rho = np.sqrt(eps)**(-1/n)
            rho_roots = np.abs(r + np.sqrt(r**2 - 1))
            rho_roots[rho_roots < 1] = 1./rho_roots[rho_roots < 1]
            r = r[rho_roots <= rho]

    elif n <= 513:
        # If n <= 513 then we can compute the new coefficients with a
        # matrix-vector product
        TLeft, TRight = fft_weights(splitPoint)

        # Compute the new coefficients
        cLeft = np.matmul(TLeft[:n, :n], c)
        cRight = np.matmul(TRight[:n, :n], c)

        # Recurse
        rLeft = roots_main(cLeft, 2*htol, splitPoint)
        rRight = roots_main(cRight, 2*htol, splitPoint)
        r = np.concatenate(((splitPoint - 1)/2 + (splitPoint + 1)/2 * rLeft,
                            (splitPoint + 1)/2 + (1 - splitPoint)/2 * rRight))

    else:
        # Otherwise split with Clenshaw
        xLeft = chebptsAB(n, [-1, splitPoint])
        xRight = chebptsAB(n, [splitPoint, 1])

        xs = np.concatenate((xLeft, xRight))
        v = clenshaw(xs, c)

        # get the coefficients on the left
        cLeft = polyfit(v[:n]).squeeze()

        # get the coefficients on the right
        cRight = polyfit(v[n:]).squeeze()

        # Recurse
        rLeft = roots_main(cLeft, 2*htol, splitPoint)
        rRight = roots_main(cRight, 2*htol, splitPoint)
        r = np.concatenate(((splitPoint - 1)/2 + (splitPoint + 1)/2*rLeft,
                            (splitPoint + 1)/2 + (1 - splitPoint)/2 * rRight))

    return np.asarray(r)

def roots_scalar(f, eps=1.48e-8):
    splitPoint = -0.004849834917525

    # Trivial case for f constant:
    if f.shape[0] == 1:
        if f.coeffs[0, 0] == 0:
            r = [0]
        else:
            r = []
        return np.asarray(r)

    # Get scaled coefficients for the recursive calls
    c = f.coeffs / f.vscale

    # call the recursive roots_main function
    r = roots_main(c.squeeze(), 100*eps, splitPoint)
    return r

def roots(f, *args, **kwargs):
    # Scalar functions
    if f.shape[1] == 1:
        r = roots_scalar(f, *args, **kwargs)
    else:
        # Need to compute the zeros per column
        r = np.empty(f.m, dtype=object)

        for i in range(f.m):
            r[i] = roots_scalar(f[i], *args, **kwargs)

    return r
