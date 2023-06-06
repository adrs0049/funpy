#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
from copy import deepcopy
from math import sqrt

import warnings
import numpy as np

from .qnerr import qnerr


def nleq_err(NewtonSystem, x, linear_solver='lu',
             tol=np.finfo(float).eps, theta_max=0.5,
             restricted=True, nonlin='mildly', inner=lambda x, y: np.dot(x, y),
             callback=lambda *args: True, dtype=float,
             precond=False, use_qnerr=True, op_kwargs={}, *args, **kwargs):
    """
        NLEQ_ERR: Nonlinear solver.
    """
    # Booleans to keep track of iteration
    linear_success = False
    newton_success = False
    lambda_limitation = False
    debug = kwargs.get('debug', False)

    assert nonlin in ['mildly', 'highly', 'extremely']

    # Default lambda values
    lambda_start       = {'mildly': 1.0, 'highly': 1.0e-2, 'extremely': 1.0e-4}
    lambda_min_default = {'mildly': 1.0e-4, 'highly': 1.0e-4, 'extremely': 1.0e-12}
    miter_default      = {'mildly': 50, 'highly': 75, 'extremely': 75}

    # Max iterations
    miter = kwargs.pop('miter', miter_default[nonlin])

    # Linear solver
    linear_solver = linear_solver.lower()

    # Setup dampening values
    lam        = kwargs.pop('lam', lambda_start[nonlin])
    lambda_min = kwargs.pop('lambda_min', lambda_min_default[nonlin])
    lam0 = lam

    # Iteration values
    k = 0
    neval = 0

    # Create norm function from inner product
    def norm(x):
        return sqrt(inner(x, np.conj(x))) if isinstance(x, np.ndarray) else sqrt(inner(x, x))

    # Setup initial norms
    mue = 0.0
    normdx = 0
    normdx_prev = 1
    normdxbar = 0

    # temporary memory
    nsystem = None
    n       = 0
    fxk     = None

    # Should we execute a Quasi-Newton error iterations?
    qnerr_iter = False

    while not newton_success:

        # check for convergence now
        if k >= miter:
            if debug: warnings.warn(
                f'NLEQ_ERR: failed to converge in {miter} iterations! Norm: {normfk:.6g}',
                RuntimeWarning)
            break

        # Compute Jacobian and RHS
        nsystem = NewtonSystem(x, exact=linear_solver, **op_kwargs)
        neval += 1

        if k > 0:
            # Recompute norms after rescaling
            normdxkm1 = norm(dx)
            normdxbar = norm(dxbar)
        else:   # if first iterate
            n   = nsystem.size
            fxk = np.empty(n, dtype=dtype)

            # Compute the right hand side
            fxk[:] = nsystem.rhs(x)
            normfk = norm(fxk)

        # Solve the linear system
        dx, linear_success = nsystem.solve(-fxk)

        # We quit if we failed to solve the linear system.
        if not linear_success:
            warnings.warn(f'NLEQ-ERR: Failed to solve linear system after {k} iterations! Norm: {normfk:.6g}', RuntimeWarning)
            newton_success = False
            break

        # descale - the newton step! Don't need to do this
        normdx = norm(dx)

        # Solution found if condition here true
        if normdx <= tol:
            x += dx
            k += 1
            newton_success = True
            break

        if k > 0:
            w = dxbar - dx
            s = norm(w)
            mue = (normdxkm1 * normdxbar) / (s * normdx) * lam if s > 0 else np.inf
            lam = min(1.0, mue)

        if debug:
            print(f'k = {k}; μ = {mue:.4g}; |fk| = {normfk:.4e}; |Δx| = {normdx:.4e}; tol = {tol:.4g}.')

        # DO ADJUST DAMPENING STEP - the while loop below adjusts that
        reducted = False

        while True:
            if lam <= lambda_min:
                # CONVERGENCE FAILURE!
                newton_success = False
                lambda_limitation = True
                break

            # New trial iterate
            xkp1 = x + lam * dx

            # Evaluate function at xkp1 - Don't regenerate the matrix!
            fxk[:] = nsystem.rhs(xkp1)
            normfkp1 = norm(fxk)

            # Solve the linear system
            dxbar, linear_success = nsystem.solve(-fxk)

            if not linear_success:
                newton_success = False
                break

            # De-scale - don't need to scale here!
            normdxbar = norm(dxbar)

            # Compute the estimator
            theta = normdxbar / normdx
            s = 1.0 - lam
            w = dxbar - s * dx
            w_norm = norm(w)
            mue = (0.5 * normdx * lam * lam) / w_norm if w_norm > 0.0 else np.inf

            if debug:
                print(f'\tk = {k}; λ = {lam:.4g}; |fk|: {normfkp1:.4e}; |Δx|: {normdxbar:.4e}; |w|: {w_norm:.4g}, θ = {theta:.6g}; μ = {mue:.4g}; s = {s:.4g}.')

            if (not restricted and theta >= 1.0) or (restricted and theta > 1.0 - lam / 4.0):
                lambda_new = min(mue, 0.5 * lam)
                if lam <= lambda_min:
                    lam = lambda_new
                else:
                    lam = max(lambda_new, lambda_min)

                reducted = True

                # Try and check whether we are good now!
                continue

            # ELSE
            lambda_new = min(1.0, mue)

            if lam == 1.0 and lambda_new == 1.0:
                if normdxbar <= tol:
                    # CONVERGED! QUIT
                    x = xkp1 + dxbar
                    newton_success = True
                    break

                qnerr_iter = (theta < theta_max)
            else:
                if lambda_new >= 4.0 * lam and not reducted:
                    lam = lambda_new

                    # Try and check again whether we are good now!
                    continue

            # If we get to the end of the loop we should quit the loop
            break

        # END OF DAMPENING ADJUSTMENT

        # Check errors that may have been triggered in the above loop!
        if not linear_success:
            warnings.warn(f'NLEQ-ERR: Failed to solve linear system after {k} iterations! Norm: {normfk:.6g}', RuntimeWarning)
            newton_success = False
            break

        if lambda_limitation:
            warnings.warn(f'NLEQ-ERR: Failed to solve nonlinear system lambda0 = {lam0};' +
                          f'(lambda = {lam:.4g} <= {lambda_min:.4g} too small) after {k+1} iterations!' +
                          f' Current norm |dx| = {normdx:.4g} > {tol:.4g}.', RuntimeWarning)
            newton_success = False
            break

        # Update last rhs norm
        normfk = normfkp1

        # NEXT STEP
        k += 1

        # callback
        normdx_prev = lam * normdx
        success_callback = callback(nsystem, k, dx, normdx_prev, theta)
        if not success_callback:
            print('NLEQ-ERR: Failed callback indicates STOP after %d iterations!' % k)
            newton_success = False
            break

        # In the case we encountered success in the previous loop
        if newton_success:
            break

        # Save previous iterate for scaling purposes and accept new iterate
        x = xkp1

        # Perform QNERR STEPS IF CHOSEN!
        if qnerr_iter and use_qnerr:
            # if qnerr_iter failed we try to continue with NLEQ_ERR!
            x, success, kp, normdx, neval_p, _ = qnerr(
                nsystem, x, itr=k, dx0=dx, tol=tol, dtype=dtype,
                inner=inner, callback=callback,
                precond=precond, theta_max=theta_max, miter=miter-k+1,
                nleqcalled=True, op_kwargs=op_kwargs, *args, **kwargs)

            # update local iteration data
            k += kp
            neval += neval_p

            if not success:
                qnerr_iter = False

            # Recompute the right hand side since the solution has changed now!
            fxk[:] = nsystem.rhs(x)
            normfk = norm(fxk)

    # END MAIN SOLVER LOOP!
    return x, newton_success, k, normdx, neval, nsystem
