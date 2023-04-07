#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
from copy import deepcopy
from math import sqrt
import warnings


def qnerr(NewtonSystem, x, tol=np.finfo(float).eps,
          dx0=None, itr=0, nleqcalled=False, miter=25,
          verb=True, debug=False, theta_max=0.5,
          callback=lambda *args: True,
          inner=lambda x, y: np.dot(x, y),
          dtype=float, op_kwargs={}, *args, **kwargs):

    """ Implements QNERR - ERRor-based Quasi-Newton algorithm

        Warning: The current implementation will not work stand-alone, but is designed to
        take input from NLEQ_ERR.

        System : represents the nonlinear system

            -> Needs to have methods to compute the nonlinear residual and the matrix at various
            points in phase space.

    """
    neval = 0

    if not nleqcalled:
        newton_success = False
        lambda_limitation = False

    # Setup the iteration
    newton_success = False

    # allocate memory
    dx    = np.empty(miter + 2, dtype=object)
    sigma = np.zeros(miter + 2)

    # Setup initial data
    k = 0
    normdx = 0
    skipstep = nleqcalled

    # Create norm function from inner product
    def norm(x):
        return sqrt(inner(x, np.conj(x))) if isinstance(x, np.ndarray) else sqrt(inner(x, x))

    # Temporary memory
    N   = -1
    fxk = None

    if not skipstep:
        nsystem = NewtonSystem(x, **op_kwargs)
        neval += 1

        N      = nsystem.size
        fxk    = np.empty(N, dtype=dtype)
        fxk[:] = nsystem.rhs(x)
        normfk = norm(fxk)

        # Call the solver
        dxk, success = nsystem.solve(-fxk)
        normdx = norm(dxk)

        # We quit if we failed to solve the linear system.
        if not success:
            if verb: warnings.warn(f'QNERR: Failed to solve linear system after {k+1} iterations!  Norm: {normfk:.6g}.',
                                   RuntimeWarning)
            newton_success = False
            if not nleqcalled: newton_success = False
            return x, newton_success, k, normdx

        # descale - the newton step! Don't need to do this
        sigma[0] = normdx
        dx[0] = dxk

        # default value
        thetak = 0.5 * theta_max

        success_callback = callback(nsystem, k, dxk, normdx, thetak)
        if not success_callback:
            if verb: warnings.warn(f'QNERR: Failed callback indicates STOP after {k} iterations!')
            newton_success = False
            return x, newton_success, k, normdx
    else:
        assert dx0 is not None, ''
        nsystem = NewtonSystem

        # Create temporary memory
        N      = nsystem.size
        fxk    = np.empty(N, dtype=dtype)

        # Compute initial quantities
        normdx = norm(dx0)
        sigma[0] = normdx**2
        dx[0] = dx0

    while not newton_success and k <= miter:
        if not skipstep:
            x += dx[k]

            if sigma[k] <= tol * tol:
                k += 1
                newton_success = True
                break

        # no longer skipping steps
        skipstep = False
        fxk[:] = nsystem.rhs(x)
        normfk = norm(fxk)
        v, success = nsystem.solve(-fxk)

        if not success:
            if verb: warnings.warn(f'QNERR: Failed to solve linear system after {k+1} iterations!',
                                   RuntimeWarning)
            newton_success = False
            break

        for i in range(1, k + 1):
            alpha_bar = inner(v, dx[i - 1]) / sigma[i - 1]
            v += alpha_bar * dx[i]

        alpha_kp1 = inner(v, dx[k]) / sigma[k]
        thetak = sqrt(inner(v, v) / sigma[k])

        if debug:
            print('\tkq = %d; α = %.4g; θ = %.4g; |v|: %.4g; |Δx|: %.4g; ε = %.2g; σ = %.2g'
                  % (k, alpha_kp1, thetak, v.norm(), dx[k].norm(), tol, sigma[k]))

        # compute new step
        s = 1.0 - alpha_kp1

        # TODO: this sometimes causes states to reshape themselves!
        dx[k + 1] = v
        dx[k + 1] /= s
        sigma[k + 1] = inner(dx[k + 1], dx[k + 1])
        k += 1

        # callback
        callback_success = callback(nsystem, itr + k, v, normdx, thetak)

        # Compute this after since the callback requires the previous normdx value
        normdx = sqrt(sigma[k])

        if thetak > theta_max or not callback_success:
            if not nleqcalled:
                if verb: warnings.warn('QNERR: Failed to solve nonlinear system after {0:d} iterations! θ = {1:.4g} greater than {2:.4g} tolerance.'
                          .format(k+1, thetak, theta_max), RuntimeWarning)
            newton_success = False
            lambda_limitation = True
            break

    # END MAIN SOLVER LOOP!
    return x, newton_success, k, normdx, neval, nsystem
