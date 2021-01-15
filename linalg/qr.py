#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
#
# Forked from the scipy repository, and extended to return the number of Householder reflections.
#
import numpy

# Local imports
from scipy.linalg.lapack import get_lapack_funcs
from scipy.linalg.misc import _datacopied


def safecall(f, name, *args, **kwargs):
    """Call a LAPACK routine, determining lwork automatically and handling
    error return values"""
    lwork = kwargs.get("lwork", None)
    if lwork in (None, -1):
        kwargs['lwork'] = -1
        ret = f(*args, **kwargs)
        kwargs['lwork'] = ret[-2][0].real.astype(numpy.int_)
    ret = f(*args, **kwargs)
    if ret[-1] < 0:
        raise ValueError("illegal value in %dth argument of internal %s"
                         % (-ret[-1], name))
    return ret[:-2]


def qr(a, overwrite_a=False, lwork=None, mode='full', pivoting=False,
       check_finite=True):
    # 'qr' was the old default, equivalent to 'full'. Neither 'full' nor
    # 'qr' are used below.
    # 'raw' is used internally by qr_multiply
    if mode not in ['full', 'qr', 'r', 'economic', 'raw']:
        raise ValueError("Mode argument should be one of ['full', 'r',"
                         "'economic', 'raw']")

    if check_finite:
        a1 = numpy.asarray_chkfinite(a)
    else:
        a1 = numpy.asarray(a)
    if len(a1.shape) != 2:
        raise ValueError("expected a 2-D array")
    M, N = a1.shape
    overwrite_a = overwrite_a or (_datacopied(a1, a))

    if pivoting:
        geqp3, = get_lapack_funcs(('geqp3',), (a1,))
        qr, jpvt, tau = safecall(geqp3, "geqp3", a1, overwrite_a=overwrite_a)
        jpvt -= 1  # geqp3 returns a 1-based index array, so subtract 1
    else:
        geqrf, = get_lapack_funcs(('geqrf',), (a1,))
        qr, tau = safecall(geqrf, "geqrf", a1, lwork=lwork,
                           overwrite_a=overwrite_a)

    if mode not in ['economic', 'raw'] or M < N:
        R = numpy.triu(qr)
    else:
        R = numpy.triu(qr[:N, :])

    if pivoting:
        Rj = R, jpvt
    else:
        Rj = R,

    if mode == 'r':
        return Rj
    elif mode == 'raw':
        return ((qr, tau),) + Rj

    gor_un_gqr, = get_lapack_funcs(('orgqr',), (qr,))

    if M < N:
        Q, = safecall(gor_un_gqr, "gorgqr/gungqr", qr[:, :M], tau,
                      lwork=lwork, overwrite_a=1)
    elif mode == 'economic':
        Q, = safecall(gor_un_gqr, "gorgqr/gungqr", qr, tau, lwork=lwork,
                      overwrite_a=1)
    else:
        t = qr.dtype.char
        qqr = numpy.empty((M, M), dtype=t)
        qqr[:, :N] = qr
        Q, = safecall(gor_un_gqr, "gorgqr/gungqr", qqr, tau, lwork=lwork,
                      overwrite_a=1)

    # Count the number of Householder reflections, so that we can calculate the determinant of Q easily!
    # http://icl.cs.utk.edu/lapack-forum/viewtopic.php?f=2&t=1741
    # All this to say that you should scan TAU from 1 to n (m > n ) or 1 to n-1 (m <= n) and count the nonzero elements. Say you get k nonzeros in TAU. Then (-1)^k.
    if M <= N:
        k = numpy.count_nonzero(tau[:-1])
    else:
        k = numpy.count_nonzero(tau)

    return (Q,) + Rj + (k,)
