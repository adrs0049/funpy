#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
import scipy
import scipy.linalg as LA
import scipy.sparse.linalg as LAS
from copy import deepcopy
from sparse.csr import eliminate_zeros_csr

from fun import Fun, h1norm, norm
from cheb.chebpts import quadwts
from cheb.diff import computeDerCoeffs
from states.State import NonlinearFunction, ContinuationState
from states.pseudo_arclength import PseudoArcContinuationCorrector
from nlep.nullspace import right_null_vector
from newton.deflated_residual import DeflatedResidual

SMALL=1.0e-150
EPMACH=1.0e-17
THETA_MAX=0.5

def scaled_norm2(v, scale):
    n = v.shape[0]
    t = v / scale
    rval = np.dot(t, t)
    return np.sqrt(rval / n)


def norm2(v):
    n = v.shape[0]
    rval = np.dot(v, v)
    return np.sqrt(rval / n)


def scale(v, scale):
    return v / scale


def descale(v, scale):
    return v * scale


def rescale(x, xa, xthres):
    return np.maximum(xthres, np.maximum(0.5 * (np.abs(x) + np.abs(xa)), SMALL))


class NewtonBase:
    def __init__(self, system, LinOp, maxiter=20, inner_maxiter=20, outer_k=10, *args, **kwargs):
        # system is a class that provides an option to solve a linear problem
        self.system = system
        self.linop = LinOp

        # For the moment simply do what scipy does when calling lgmres for
        # Newton's method
        self.method = scipy.sparse.linalg.lgmres
        # TODO: create a preconditioner
        self.method_kw = dict(maxiter=inner_maxiter)
        self.method_kw['outer_k'] = outer_k
        self.method_kw['maxiter'] = maxiter
        # Carry LGMRES vectors across nonlinear iterations
        self.method_kw['outer_v'] = []
        self.method_kw['prepend_outer_v'] = True
        # We don't store the Jacobian * v products, in case they change a lot
        # in the nonlinear step!
        self.method_kw['store_outer_Av'] = False
        self.method_kw['atol'] = 0

    def to_state(self, coeffs):
        """ Constructs a new state from coefficients """
        soln = Fun(coeffs=coeffs[:-1], domain=self.system.domain,
                   simplify=False, type='cheb')
        return ContinuationState(a=coeffs[-1], u=soln, n=coeffs.size-1)

    def isolve(self, LinOp, rhs, x0, **kwargs):
        # TODO: somehow save temporary lmgres results to speed up convergence
        # between separate call to the lgmres solver!
        coeffs, info = LAS.lgmres(LinOp, rhs, x0, **kwargs)

        # check that the solution was truly solved
        success = np.allclose(LinOp.matvec(coeffs), rhs, atol=1e-6, rtol=1e-7) and info == 0
        res = np.max(np.abs(LinOp.matvec(coeffs) - rhs))

        if info > 0:
            print('LGMRES: %.4g did not converge!' % res)
        elif info < 0:
            print('LGMRES: %.4g received illegal input or breakdown!' % res)
        elif not success:
            print('LGMRES: %.4g claims to have worked but residual check failed!' % res)

        # construct a new state
        return self.to_state(coeffs), success

    def newton_step(self, LinOp, un, miter=25, *args, **kwargs):
        ###############################
        # Step 2: Corrector           #
        ###############################
        iteration = 0
        newton_success = False
        inner_tol = kwargs.pop('inner_tol', 1e-8)

        # This is the step we are solving for!
        du = np.zeros_like(un)

        while not newton_success:
            # check for convergence now
            if iteration > miter:
                break

            # Assemble the new inner linear system
            linOp = LinOp(un)
            print(linOp.todense())

            # Finally call LGMRES
            du, success = self.isolve(linOp, -linOp.b, np.zeros_like(un), tol=inner_tol)

            # print('a:', un.a, ' res:', LA.norm(LinOp.b))
            # print('du:', du.u.values.T)
            # print('un:', un.u.values.T)

            if not success:
                print('Failed to solve the linear system! Quitting...')
                break

            if du.norm() > 1e4:
                print('||du|| = %.4g! That is too large!' % du.norm())
                break

            # this is here so that we return the correct residual
            if du.norm() < tol:
                newton_success = True

            # update the solution
            un += du

            # increment iteration
            iteration += 1

        return un, newton_success, iteration

    def nleq_err(self, LinOp, x, miter=25, *args, **kwargs):
        tol = kwargs.pop('tol', 1e-8)
        inner_tol = kwargs.pop('inner_tol', 1e-8)

        # Setup the iteration
        iteration = 0
        newton_success = False
        lambda_min = 1e-8
        lam = 1.0
        k = 0

        # Setup initial norms
        normdx = 0
        normdxbar = 0
        w = np.copy(x)
        xscale = kwargs.pop('xscale', None)
        if xscale is None:
            xscale = np.ones_like(x)
        xthresh = np.copy(xscale)

        while not newton_success:
            # check for convergence now
            if iteration > miter:
                print('Deflated newton method failed to converge in %d iterations!' % miter)
                break

            # rescale
            xscale = rescale(x, w, xthresh)

            if k > 0:
                # Recompute norms after rescaling
                normdxkm1 = scaled_norm2(dx, xscale)
                normdxbar = scaled_norm2(dxbar, xscale)

            # Compute jacobian and RHS -
            linOp = LinOp(x)

            # Scale jacobian & switch the sign of the jacobian?
            # Do we need to scale the jacobian?
            # TODO: Jacobian scaling!
            normfk = norm2(linOp.b)

            # Compute Newton correction -> uses residual as IC
            # Can also equivalently flip the sign of the RHS
            dx, success = self.isolve(linOp, -linOp.b, np.zeros(state.n+1),
                                      tol=inner_tol)

            # descale - the newton step!
            dx = descale(dx, xscale)
            normdx = scaled_norm2(dx, xscale)

            # Solution found if condition here true
            if normdx <= tol:
                x += dx
                k += 1
                newton_success = True
                break

            if k > 0:
                w = dxbar - dx
                s = scaled_norm2(w, xscale)
                mue = (normdxkm1 * normdxbar)/(s * normdx) * lam
                self.lam = min(1.0, mue)

            # DO ADJUST DAMPENING STEP - the while loop below adjusts that
            reducted = False

            while True:
                if lam < lambda_min:
                    # CONVERGENCE FAILURE!
                    newton_success = False
                    break

                # New trial iterate
                xkp1 = x + self.lam * dx

                # Evaluate function at xkp1 - Don't regenerate the matrix!
                fxk = linOp.rhs(xkp1)
                normfkp1 = norm2(fxk)

                # Compute Newton correction -> uses residual as IC
                # Can also equivalently flip the sign of the RHS
                dxbar, success = self.isolve(LinOp, -fxk, np.zeros(state.n+1),
                                             tol=inner_tol)

                # De-scale
                dxbar = descale(dxbar, xscale)
                normdxbar = scaled_norm2(dxbar, xscale)

                # Compute the estimator
                theta = normdxbar / normdx
                s = 1.0 - self.lam
                w = dxbar - s * dx
                w_norm = scaled_norm2(w, xscale)
                mue = (0.5 * normdx * self.lam * self.lam) / w_norm

                if theta >= 1.0:
                    lambda_new = min(mue, 0.5 * self.lam)
                    if lam < lambda_min:
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

                    # TODO: this
                    qnerr_iter = (theta < THETA_MAX)
                else:
                    if lambda_new >= 4.0 * lam and not reducted:
                        lam = lambda_new

                        # Try and check again whether we are good now!
                        continue

            # END OF DAMPENING ADJUSTMENT

            # Save previous iterate for scaling purposes and accept new iterate
            w = np.copy(x)
            x = np.copy(xkp1)

            # NEXT STEP
            k += 1
            normfk = mormfkp1

            # Perform QNERR STEPS IF CHOSEN!
            if qnerr_iter:
                assert False, 'Requested a qnerr_iter!'

        # END MAIN SOLVER LOOP!
