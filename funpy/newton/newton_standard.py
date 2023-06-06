#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
from math import sqrt
import numpy as np


def newton_standard(NewtonSystem, x, linear_solver='lu',
                    tol=np.finfo(float).eps, miter=50, inner=lambda x, y: np.dot(x, y),
                    callback=lambda *args: True, dtype=float,
                    precond=False, *args, **kwargs):

    # Booleans to keep track of iteration
    linear_success = False
    newton_success = False
    debug = kwargs.get('debug', False)

    # Linear solver
    linear_solver = linear_solver.lower()

    # Iteration values
    k = 0
    neval = 0

    # Create norm function from inner product
    def norm(x):
        return sqrt(inner(x, np.conj(x))) if isinstance(x, np.ndarray) else sqrt(inner(x, x))

    # Setup initial norms
    normdx = 0

    # temporary memory
    nsystem = None
    n       = 0
    fxk     = None

    while not newton_success:

        # check for convergence now
        if k >= miter:
            if debug: warnings.warn(
                f'NEWTON: failed to converge in {miter} iterations! Norm: {normfk:.6g}',
                RuntimeWarning)
            break

        # Compute Jacobian and RHS
        nsystem = NewtonSystem(x, exact=linear_solver)
        neval += 1

        if k == 0:
            n   = nsystem.size
            fxk = np.empty(n, dtype=dtype)

        # Compute the right hand side
        fxk[:] = nsystem.rhs(x)
        normfk = norm(fxk)

        # Solve the linear system
        dx, linear_success = nsystem.solve(-fxk)

        # We quit if we failed to solve the linear system.
        if not linear_success:
            warnings.warn(f'NEWTON: Failed to solve linear system after {k} iterations! Norm: {normfk:.6g}', RuntimeWarning)
            newton_success = False
            break

        # descale - the newton step! Don't need to do this
        normdx = norm(dx)

        # NEXT STEP
        x += dx
        k += 1

        if debug:
            print(f'k = {k}; |fk| = {normfk:.4e}; |Δx| = {normdx:.4e}; tol = {tol:.4g}.')

        # Solution found if condition here true
        if normdx <= tol:
            newton_success = True
            break

        # callback
        success_callback = callback(nsystem, k, dx, normdx, 0.0)
        if not success_callback:
            print('NEWTON: Failed callback indicates STOP after %d iterations!' % k)
            newton_success = False
            break

    # END MAIN SOLVER LOOP!
    return x, newton_success, k, normdx, neval, nsystem
