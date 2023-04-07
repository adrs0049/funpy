#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
from enum import Enum
from math import sqrt
import warnings
from copy import deepcopy
import numpy as np

from linalg.gbit import gbit


class Adaptmode(Enum):
    STANDARD = 0
    QUADRATIC = 1


def giant_gbit(NewtonSystem, x, tol=1e6 * np.finfo(float).eps,
               nonlin='mildly', max_iter=50,
               adaptmode=Adaptmode.QUADRATIC,
               inner=lambda x, y: np.inner(x, y),
               rescale=False, precision=np.finfo(float).eps,
               dtype=float, delta_bar=1.0e-3, *args, **kwargs):
    """
    GIANT-GBIT: Nonlinear solver.
    """
    assert nonlin in ['mildly', 'highly', 'extremely']

    # Default lambda values
    lambda_start       = {'mildly': 1.0, 'highly': 1.0e-2, 'extremely': 1.0e-4}
    lambda_min_default = {'mildly': 1.0e-4, 'highly': 1.0e-4, 'extremely': 1.0e-8}

    debug      = kwargs.pop('debug', False)
    xtol       = tol
    lam        = lambda_start[nonlin]
    lambda_min = lambda_min_default[nonlin]
    restricted = True if nonlin == 'extremely' else False
    safetyfactor = kwargs.pop('safetyfactor', 0.1)
    if safetyfactor <= 0.0 or safetyfactor > 1.0: safetyfactor = 1.0
    rho = safetyfactor
    rhotilde = 0.5 * rho
    rhobar_max = rhotilde / (1.0 + rhotilde)

    # Create norm function from inner product
    def norm(x):
        return sqrt(inner(x, np.conj(x))) if isinstance(x, np.ndarray) else sqrt(inner(x, x))

    if debug:
        print(f'Solver tolerance {xtol:.2g}')

    # Counters
    nfcn = 0
    njac = 0
    liniter = 0
    nmuljac = 0
    nprecon = 0
    nordlin = 0
    nsimlin = 0

    # storage requirement
    n = np.product(x.shape)

    # Allocate memory
    dx      = np.zeros_like(x)
    dxbar   = np.zeros_like(x)
    fxk     = np.empty(n, dtype=dtype)

    # Temp variables
    hk        = 0.0
    hkpri     = 0.0
    normdx    = 0.0
    normdxbar = 0.0

    # Success tracker
    newton_success = False

    # Main iteration loop
    k = 0
    rcode = 2
    while k <= max_iter and rcode == 2:
        # Do something with jac? -> Compute jacobian
        nsystem = NewtonSystem(x)
        linOp, precd, proj = nsystem.matrix(precond=True)
        njac += 1

        if k > 0:
            normdxkm1 = norm(dx)
            normdxbar = norm(dxbar)
        else:  # Only need to do this here when k == 0
            fxk[:] = nsystem.rhs(x)
            normfk = norm(fxk)

        # Call the linear solver
        tol = 0.25 if adaptmode == Adaptmode.QUADRATIC else rho / (2.0 * (1.0 + rho))
        dx, success, iter, nomatvec, noprecon = gbit(
            linOp, precd, -fxk, y=dxbar, rho=rho, to_fun=nsystem.to_vspace,
            inner=inner, tol=tol, dtype=dtype, giant_simple=False, **kwargs)

        # Update these from the linear solver
        liniter += iter
        nmuljac += nomatvec
        nprecon += noprecon

        if not success:
            warnings.warn('GIANT linear system solve failed!')
            newton_success = False
            break

        # Start with the step adaptations
        normdx = norm(dx)

        if debug:
            print(f'\tk = {k}; iter = {iter}; dx = {normdx:.2g}; normfk = {normfk:.2g}')

        if k > 0:
            hkpri = normdx / normdxkm1 * hk
            lam = min(1.0, 1.0 / ((1.0 + rho) * hkpri))

        if lam == 1.0 and k > 0:
            tol = rho / 2.0 * hk / (1.0 + hk) if adaptmode == Adaptmode.QUADRATIC else delta_bar
            dx, success, iter, nomatvec, noprecon = gbit(
                linOp, precd, -fxk, y=dx, inner=inner, to_fun=nsystem.to_vspace,
                dtype=dtype, tol=tol, rho=rho, giant_simple=False, **kwargs)

            liniter += iter
            nmuljac += nomatvec
            nprecon += noprecon

            if not success:
                warnings.warn('GIANT linear system solve failed!')
                newton_success = False
                break

            normdx = norm(dx)
            hkpri = normdx / normdxkm1 * hk
            lam = min(1.0, 1.0 / ((1.0 + rho) * hkpri))

            if debug:
                print(f'\tkq = {k}; lam = {lam:.2g}; iter = {iter}; dx = {normdx:.2g}; normfk = {normfk:.2g}')

        nordlin += liniter
        hk = hkpri

        # Check for convergence
        if normdx <= xtol:
            newton_success = True
            precision = normdx
            x += dx
            k += 1
            break

        # Iterate adaptation loop!
        reducted = False
        while True:
            if lam < lambda_min:
                warnings.warn('GIANT lambda below tolerance value!')
                newton_success = False
                return x, newton_success, k, njac, nsystem

            # Compute new trial iterate
            xkp1 = x + lam * dx
            fxk[:] = nsystem.rhs(xkp1)
            normfk = norm(fxk)
            nfcn += 1

            s = 1.0 - lam
            dxbar, success, iter, nomatvec, noprecon = gbit(
                linOp, precd, -fxk, y=s*dx, inner=inner, to_fun=nsystem.to_vspace,
                dtype=dtype, rho=rho, tol=rhobar_max, giant_simple=True, **kwargs)

            nsimlin += iter
            nmuljac += nomatvec
            nprecon += noprecon

            if not success:
                warnings.warn('GIANT linear system solve failed!')
                newton_success = False
                return x, newton_success, k, njac, nsystem

            normdxbar = norm(dxbar)
            theta = normdxbar / normdx
            w = dxbar - s * dx
            normdiff = norm(w)
            rhobar = precision * safetyfactor * normdxbar / normdiff
            hk = 2.0 * (1.0 - rhobar) * normdiff / (lam * lam * normdx)

            if debug:
                print(f'\t\tlam = {lam:.2g}; s = {s:.2f}; θ = {theta:.2g}; w = {normdiff:.2g}; iter = {iter}; normdxbar = {normdxbar:.2g}; normfk = {normfk:.2g}.')

            if (not restricted and theta >= 1.0) or (restricted and theta > 1.0 - 0.25 * lam):
                lam_new = min(1.0 / (hk * (1.0 + rho)), 0.5 * lam)
                lam = lam_new if lam <= lambda_min else max(lam_new, lambda_min)
                reducted = True

                # GO CHECK REG AGAIN!
                continue

            lam_new = min(1.0, 1.0 / (hk * (1.0 + rho)))
            if lam == 1.0 and lam_new == 1.0:
                if normdxbar <= xtol:
                    x += dxbar
                    precision = normdxbar
                    # RETURN SUCCESS
                    rcode = 0
                    newton_success = True
                    break
            else:
                if lam_new >= 4.0 * lam and not reducted:
                    lam = lam_new

                    # GO CHECK REG AGAIN!
                    continue

            # If we have gotten this far we can quit the adaptation loop!
            break

        # END OF ADJUSTMENT LOOP
        k += 1

        # If we encountered success in the adjustment loop make sure we break the outside loop
        if newton_success:
            break

        # FINAL RETURN NEW ACCEPTED ITERATE
        x = xkp1

    # MAIN LOOP DONE
    return x, newton_success, k, njac, nsystem
