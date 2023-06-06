#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
from math import sqrt
from copy import deepcopy
import numpy as np


def isqrt(x):
    return sqrt(abs(x))


def gbit(A, P, b, y=None,
         max_iter=2000, i_max=10, rho=1e-3,
         tol=np.finfo(float).eps,
         giant_simple=False, dtype=float,
         tau_min=1.0e-8, tau_max=1.0e2, epsilon_limit=1.0e20,
         to_fun=lambda x: x,
         inner=lambda x, y: np.dot(x, y), **kwargs):
    """

        Arguments:
            A  - square linear system
            P  - preconditioner
            b  - The system's right hand side
            y - initial solution guess
    """
    assert A.shape[0] == A.shape[1], 'Matrix must be square!'
    assert rho <= 1.0, ''
    n = A.shape[0]
    if y is None: y = np.zeros(n)

    # counters
    iter = 0
    nomatvec = 0
    noprecon = 0

    # tolerances
    errtol = tol

    # Create norm function from inner product
    def norm(x):
        return sqrt(inner(x, x))

    # Allocate various required temporary arrays
    delta   = np.empty(i_max + 2, dtype=object)
    sigma   = np.empty(i_max + 2, dtype=dtype)
    t       = np.empty(i_max + 2, dtype=dtype)
    q       = np.empty(n, dtype=dtype)
    zeta    = None

    # copy initial guess
    y0 = deepcopy(y)

    # MAIN iteration loop!
    success = False
    restart = True

    while restart:
        # Initialization
        z = b - A * np.asarray(y)
        nomatvec += 1

        delta[0] = to_fun(P * z)
        noprecon += 1

        sigma[0] = inner(delta[0], delta[0])

        # Main iteration loop
        normyip1 = norm(y)

        # INNER LOOP
        i = 0
        stop_iter = False
        while i <= i_max and not stop_iter and iter <= max_iter:
            q[:] = A * np.asarray(delta[i])
            nomatvec += 1

            zeta = to_fun(P * q)
            noprecon += 1

            # Update loop m = 0, ..., i - 1 (for i >= 1)
            for m in range(i):
                tm1 = 1.0 - t[m]
                fac = inner(delta[m], zeta) / sigma[m]
                zeta += fac * (delta[m+1] - tm1 * delta[m])

            # continue
            gamma = inner(delta[i], zeta)
            tau = sigma[i] / gamma if gamma != 0.0 else 2.0 * tau_max

            if tau < tau_min:
                if i > 0:  # RESTART
                    break
                else:
                    t[i] = 1.0
            else:
                t[i] = tau if tau <= tau_max else 1.0

            # update estimate
            ti = t[i]
            y += ti * delta[i]

            # Prepare for next step
            fac = 1.0 - ti + tau
            delta[i+1] = fac * delta[i] - tau * zeta
            sigma[i+1] = inner(delta[i+1], delta[i+1])
            epsilon = 0.5 * isqrt(sigma[i-1] + 2.0 * sigma[i] + sigma[i + 1]) if i >= 1 else isqrt(sigma[1])
            normyip1 = norm(y)

            if giant_simple:
                normdiff = norm(y - y0)
                stop_iter = False if normdiff == 0.0 else (epsilon / normdiff <= rho * errtol)
            else:
                stop_iter = epsilon <= rho * normyip1 * errtol

            # Update iteration counters
            iter += 1
            i += 1

            if not stop_iter and epsilon > epsilon_limit:
                success = False
                # Make sure outside loop quits!
                stop_iter = False
                break

        # END INNER LOOP
        if stop_iter or iter >= max_iter:
            break

    # LOOP END
    success = stop_iter & (iter < max_iter)
    return y, success, iter, nomatvec, noprecon


if __name__ == '__main__':
    from scipy.sparse import csr_matrix
    a = np.array([[1, 2], [3, 5]])
    b = np.array([1, 2])
    x = np.linalg.solve(a, b)
    print('expected = ', x)
    np.allclose(np.dot(a, x), b)

    # Test gbit
    a = np.array([[1, 2], [3, 5]])
    p = np.eye(2)
    a = csr_matrix(a)
    p = csr_matrix(p)

    x, success, iter, nomatvec, noprecon = gbit(a, p, b)
    print('x = ', x)
    print(np.allclose(a * x, b))
