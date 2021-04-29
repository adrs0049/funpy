#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import time
import numpy as np
import scipy
import scipy.linalg as LA
import scipy.sparse.linalg as LAS
import scipy.sparse as sps
from math import sqrt
import warnings
from copy import deepcopy

from fun import Fun, h1norm, norm, norm2
from fun import minandmax
from cheb.chebpts import quadwts
from cheb.diff import computeDerCoeffs
from states.State import ContinuationState
from states.deflation_state import DeflationState
from newton.pseudo_arclength import PseudoArcContinuationCorrector
from newton.newton_gauss import NewtonGaussContinuationCorrector
from newton.deflated_residual import DeflatedResidual
from linalg.qr_solve import QRCholesky
from bif.LinearSystem import LinearSystem
from nlep.nullspace import right_null_vector
from support.tools import orientation_y, logdet, functional, Determinant
from newton.solver_norms import sprod, nnorm, rescale

from bif.fold_solver import AugmentedFoldSolver

try:
    from scikits import umfpack
except ModuleNotFoundError:
    print('umfpack not installed!')


# TODO MOVE ME!
THETA_MAX = 0.5


class NewtonBase:
    def __init__(self, system, LinOp, method='qrchol',
                 maxiter=20, inner_maxiter=200, outer_k=30, damp=0.0, *args, **kwargs):
        """
        Arguments:

            method: The method by which to solve the linear system.
                Currently supported: - SPQR - sparse QR solver from suitesparse
                                       SPSOLVE - using UMFPACK
                                       LGMRES
                                       GCROTMK

            inner_maxiter: The maximum iterations for the linear system solver. If the chosen
            solver is iterative.

            maxiter: The maximum number of Newton iterations.

        """
        # system is a class that provides an option to solve a linear problem
        self.system = system
        self.linop = LinOp
        self.angle = 0.0
        self.Pr = None

        self.iterates = []
        self.steps = []
        self.debug = kwargs.pop('debug', False)
        self.verb = kwargs.pop('verb', False)
        if self.debug: print('Newton using %s.' % method)

        # the shape of the solution vector
        self.shape = None
        self.cpar = 'dummy'
        self.cond = 0.0

        # function type
        self.function_type = 'cheb'

        # residual norm
        self.normfk = np.inf

        # the main matrices
        self.linOp = None
        self.precd = None

        # matrix decomp
        self.lu = None
        self.qrchol = None
        self.det = None

        # Keep track of whether the system becomes singular
        self.lambda_limitation = False
        self.solve_inner_failure = False
        self.singular = False
        self.rank = 0
        self.rank_deficiency = 0
        self.rank_deficiency_rel = 1
        self.thetak = 0.5 * THETA_MAX

        # counter the evaluations of the Frechet derivative
        self.neval = 0

        # tolerance of newton's method - and tolerance of the internal linear solver
        self.ntol = kwargs.pop('tol', 1e-12)
        self.ltol = kwargs.pop('ltol', 1e-12)

        # Select the numerical method
        self.isolve_default = dict(lgmres=self.isolve_lgmres,
                                   spqr=self.isolve_spqr,
                                   lsmr=self.isolve_lsmr,
                                   qrchol=self.isolve_qrchol,
                                   #fold=self.isolve_fold,
                                   fold=self.isolve_qrchol,  # OLD REMOVE ONCE FIXED!
                                   spsolve=self.isolve_spsolve,
                                   gcrotmk=self.isolve_gcrotmk,
                                  ).get(method, method)

        self.method = dict(lgmres=scipy.sparse.linalg.lgmres,
                           spqr=None,
                           lsmr=scipy.sparse.linalg.lsmr,
                           spsolve=scipy.sparse.linalg.spsolve,
                           gcrotmk=scipy.sparse.linalg.gcrotmk,
                          ).get(method, method)

        self.req_precd = dict(lgmres=True,
                           spqr=False,
                           qrchol=True,
                           fold=True,
                           lsmr=True,
                           spsolve=False,
                           gcrotmk=True,
                          ).get(method, method)

        self.method_kw = dict(maxiter=inner_maxiter)

        print('Newton = ', self.method)

        # For the moment simply do what scipy does when calling lgmres for
        # Newton's method
        if self.method is scipy.sparse.linalg.lgmres:
            self.method_kw['outer_k'] = outer_k
            self.method_kw['maxiter'] = inner_maxiter
            # Carry LGMRES vectors across nonlinear iterations
            self.method_kw['outer_v'] = []
            self.method_kw['prepend_outer_v'] = True
            # We don't store the Jacobian * v products, in case they change a lot
            # in the nonlinear step!
            self.method_kw['store_outer_Av'] = False
            self.method_kw['atol'] = 0
            self.method_kw['tol'] = self.ltol
        elif self.method is scipy.sparse.linalg.gcrotmk:
            self.method_kw['m'] = outer_k
            self.method_kw['k'] = kwargs.get('k', self.method_kw['m'])
            self.method_kw['maxiter'] = inner_maxiter
            # Carry LGMRES vectors across nonlinear iterations
            self.method_kw['truncate'] = 'oldest'
            self.method_kw['CU'] = []
            # We don't store the Jacobian * v products, in case they change a lot
            # in the nonlinear step!
            self.method_kw['discard_C'] = True
            self.method_kw['atol'] = 0
            self.method_kw['tol'] = self.ltol
        elif self.method is scipy.sparse.linalg.lsmr:
            self.method_kw['maxiter'] = inner_maxiter

    @property
    def dtype(self):
        if self.function_type == 'trig':
            return np.complex128
        return np.float64

    def setDisc(self, n):
        self.system.setDisc(n)
        self.cleanup()

    def cleanup(self):
        """ Execute clean-ups depending on what linear method we use! """
        if self.method is scipy.sparse.linalg.lgmres:
            # Reset the stored outer_v values
            self.method_kw['outer_v'] = []
        elif self.method is scipy.sparse.linalg.gcrotmk:
            # Reset the stored outer_v values
            self.method_kw['CU'] = []

        # Cleanup stuff
        self.qrchol = None
        self.Pr = None

    def to_fun(self, coeffs, ishappy=True):
        """ TODO: the collocator should do this! Map coefficients to function """
        # If we have a projection defined -> apply it.
        if self.Pr is not None:
            coeffs = self.Pr.dot(coeffs)

        # Create the new function object
        m = coeffs.size // self.n_eqn
        return Fun(coeffs=coeffs.reshape((m, self.n_eqn), order='F'),
                   simplify=False, eps=np.finfo(float).eps,
                   domain=self.system.domain,
                   ishappy=ishappy, type=self.function_type)

    def to_state(self, coeffs):
        return NotImplemented

    def isolve(self, rhs, *args, **kwargs):
        return self.isolve_default(rhs, *args, **kwargs)

    def isolve_lsmr(self, rhs, x0=None, *args, **kwargs):
        success = False
        coeffs, istop, itn, normr, normar, norma, conda, normx = LAS.lsmr(self.linOp, rhs,
                                                                          atol=self.ltol, x0=x0,
                                                                          btol=self.ltol,
                                                                          show=False, conlim=0,
                                                                          **self.method_kw)
        print('istop = ', istop)
        success = not (istop == 7)

        # Use the QR decomposition to solve the problem now
        coeffs = self.linOp.prcond._matvec(coeffs)

        if not success:
            self.solve_inner_failure = True
            success = False
            #self.cleanup()

            warnings.warn('LSMR({0:d}; {1:s}): info = {2:d}, tol = {3:.4g}, res = {4:.4g}; K(a) = {5:.4g} did not converge!'.\
                          format(itn, str(self.shape), istop, self.ltol, normr, conda),
                          RuntimeWarning)

        # construct a new state
        return self.to_state(coeffs), success

    def isolve_lgmres(self, rhs, x0=None, *args, **kwargs):
        success = False
        coeffs, info = LAS.lgmres(self.linOp, rhs, x0, M=self.precd, **self.method_kw)
        success = (info == 0)

        if info > 0:
            self.solve_inner_failure = True
            success = False
            self.cleanup()

            warnings.warn('LGMRES({0:d}; {1:s}): info = {2:d}, tol = {3:.4g}, did not converge!'.\
                          format(self.method_kw['outer_k'], str(self.shape), info, self.ltol),
                          RuntimeWarning)

        elif info < 0:
            warnings.warn('LGMRES({0:d}): received illegal input or breakdown!'.\
                          format(self.method_kw['outer_k']), RuntimeWarning)

        # construct a new state
        return self.to_state(coeffs), success

    def isolve_gcrotmk(self, rhs, x0=None, *args, **kwargs):
        coeffs, info = LAS.gcrotmk(self.linOp, rhs, x0, M=self.precd, **self.method_kw)
        success, res = self.residual(coeffs, rhs, info=info)

        if info > 0:
            self.solve_inner_failure = True
            self.cleanup()
            print('GCROT(%d, %d): ||res|| = %.4g; info = %d, tol = %.4g, did not converge!' %
                  (self.method_kw['m'], self.method_kw['k'], res, info, self.ltol))

            # Solve using sparse QR
            # return self.isolve_spqr(rhs, *args, **kwargs)
        elif not success:
            print('GCROT(%d, %d): ||res|| = %.4g claims to have worked but residual check failed!' %
                  (self.method_kw['m'], self.method_kw['k'], res))

        assert np.all(np.isfinite(coeffs)), 'GCROT produced invalid result!'
        # construct a new state
        return self.to_state(coeffs), success

    def isolve_spqr(self, rhs, *args, **kwargs):
        import sparseqr
        # A = self.linOp.tosparse()
        A = self.linOp.to_matrix()
        coeffs = sparseqr.solve(A, rhs)

        # check whether we found a solution!
        if coeffs is not None:
            success, res = self.residual(coeffs, rhs, info=0)
        else:
            success = False
            res = np.inf

        if not success:
            print('SPQR: ||res|| = %.4g; tol = %.4g, did not find a solution!' % (res, self.ltol))

        return self.to_state(coeffs), success

    def isolve_fold(self, rhs, *args, **kwargs):
        if self.qrchol is None:
            self.qrchol = AugmentedFoldSolver(self.linOp)

        # solve the system
        coeffs = self.qrchol.solve(rhs)

        # check whether we found a solution!
        if coeffs is not None:
            # TODO: Fix this; currently this uses the old matrix to compute the residual!
            success, res = self.residual(coeffs, rhs, info=0, atol=1e-2, rtol=1e-3)
        else:
            success = False
            res = np.inf

        # FORCE THIS FOR THE MOMENT!
        success = True

        if not success:
            print('QRCholeskyFold: ||res|| = {0:.4g}; tol = {1:.4g}; Did not find a solution!'.
                  format(res, self.ltol))

        return self.to_state(coeffs), success

    def isolve_qrchol(self, rhs, *args, **kwargs):
        eps = 1e-14
        if self.qrchol is None:
            self.B = self.linOp.to_matrix()
            self.Pr = self.linOp.proj

            # This is P-inverse; where P = diag(A)
            self.P = self.precd.to_matrix()
            self.A = (self.B * self.P).todense()

            # Compute the QR-decomposition
            self.qrchol = QRCholesky(self.A, eps=eps, rank=rhs.size, *args, **kwargs)

            # Since we are using a rank revealing QR we can make some inference of how badly
            self.singular = self.qrchol.is_singular
            self.cond = self.qrchol.cond
            self.rank = self.qrchol.rank
            self.rank_deficiency = self.qrchol.rank_deficiency
            self.rank_deficiency_rel = self.qrchol.rank_deficiency_percent

        # Use the QR decomposition to solve the problem now
        coeffs = self.qrchol.solve(rhs)
        coeffs = self.P.dot(coeffs)
        return self.to_state(coeffs), True

    def isolve_spsolve(self, rhs, *args, **kwargs):
        if self.lu is None:
            self.A = self.linOp.to_matrix().tocsc()
            self.lu = umfpack.splu(self.A)

            # compute determinant
            self.det = Determinant(lu=self.lu)

        # Use the LU decomposition to solve the problem now
        coeffs = self.lu.solve(rhs)

        # coeffs2 = LAS.spsolve(A, rhs, use_umfpack=True)
        # diff = np.linalg.norm(coeffs1 - coeffs2, ord=2)
        # print('coeffs SPS = ', coeffs2[:10], ' diff = %.4g' % diff)

        # check whether we found a solution!
        if coeffs is not None:
            success, res = self.residual(coeffs, rhs, info=0)
        else:
            success = False
            res = np.inf

        if not success:
            sign, logdet = self.det()
            print('SPSolve: ||res|| = %.4g; tol = %.4g; logdet = (%.1f, %.4g); did not find a solution!' % (res, self.ltol, sign, logdet))

        return self.to_state(coeffs), success

    def detFx(self, *args, **kwargs):
        self.Fx = self.A[:-1, :-1]
        lu = umfpack.splu(self.Fx)

        # compute determinant
        return Determinant(lu=lu)

    def newton_step(self, LinOp, x, miter=25, *args, **kwargs):
        """ This function implements a basic newton method """
        k = 0
        self.neval = 0
        tol = self.ntol
        lam = kwargs.get('lam', 1.0)
        precond = kwargs.get('precond', self.req_precd)

        # Setup the iteration
        self.newton_success = False
        self.convergence_failure = False
        self.lambda_limitation = False
        self.solve_inner_failure = False
        self.callback_failure = False

        while not self.newton_success and not self.convergence_failure:
            # check for convergence now
            if k >= miter:
                if self.debug: print('NEWTON: failed to converge in %d iterations!' % miter)
                break

            # Assemble the new inner linear system
            self.lu = None
            self.qrchol = None
            self.linOp = LinOp(x)
            self.neval += 1
            self.precd = self.linOp.precond() if precond else None
            self.normfk = self.to_state(self.linOp.b).norm()

            # Finally call linear system solver
            dx, success = self.isolve(self.linOp.b)

            # Compute norm of the newton step
            normdx = nnorm(dx)

            # We quit if we failed to solve the linear system.
            if not success:
                warnings.warn('NEW-STD: Failed to solve linear system after %d iterations! Norm: %.6g' \
                              % (k, self.normfk), RuntimeWarning)
                self.newton_success = False
                break

            # check callback
            callback_success = self.callback(k, lam * dx, lam * normdx, 0)

            if not callback_success:
                print('NEW-STD: Failed callback indicates STOP after %d iterations!' % k)
                self.callback_failure = True
                self.newton_success = False
                break

            # update the solution
            x -= lam * dx

            # increment iteration
            k += 1

            # Solution found if condition here true
            if normdx <= tol:
                if self.debug: self.steps.append(deepcopy(dx))
                self.newton_success = True
                break

        return x, self.newton_success, k

    def callback(self, itr, dx, normdx, thetak):
        return True

    def nleq_err(self, LinOp, x, miter=25, restricted=True, scale=False, *args, **kwargs):
        self.neval = 0
        tol = self.ntol
        precond = kwargs.get('precond', self.req_precd)

        #if 1e1 * self.ltol >= tol:
        #    warnings.warn("NLEQ_ERR: The tolerance for the linear system ({0:.4g}) is larger or close to the".format(self.ltol)
        #                  + " nonlinear solver tolerance ({0:.4g}). This makes convergence difficult!".format(tol), RuntimeWarning)

        # Setup the iteration
        self.newton_success = False
        self.convergence_failure = False
        self.lambda_limitation = False
        self.solve_inner_failure = False
        self.callback_failure = False
        lambda_min = kwargs.get('lambda_min', 1e-4)
        lam = kwargs.get('lam', 1.0)
        lam0 = lam
        k = 0

        # Setup initial norms
        normdx = 0
        normdx_prev = 1
        normdxbar = 0

        # Scaling support
        xscale = None
        if scale:
            xscale = np.ones_like(x)
            xthresh = xscale
            w = x

        # Should we execute a Quasi-Newton error iterations?
        qnerr_iter = False

        while not self.newton_success and not self.convergence_failure:
            if scale:
                xscale = rescale(x, w, xthresh)

            # check for convergence now
            if k >= miter:
                if self.debug: warnings.warn('NLEQ_ERR: failed to converge in %d iterations! Norm: %.6g' % \
                                             (miter, self.normfk), RuntimeWarning)
                break

            if k > 0:
                # Recompute norms after rescaling
                normdxkm1 = nnorm(dx, xscale)
                normdxbar = nnorm(dxbar, xscale)

            # Compute Jacobian and RHS
            self.lu = None
            self.qrchol = None
            self.linOp = LinOp(x)
            self.neval += 1
            self.precd = self.linOp.precond() if precond else None
            self.normfk = self.to_state(self.linOp.b).norm()

            # Call the solver
            dx, success = self.isolve(-self.linOp.b)

            # We quit if we failed to solve the linear system.
            if not success:
                warnings.warn('NLEQ-ERR: Failed to solve linear system after %d iterations! Norm: %.6g' \
                              % (k, self.normfk), RuntimeWarning)
                self.newton_success = False
                break

            # descale - the newton step! Don't need to do this
            normdx = nnorm(dx, xscale)

            if self.debug:
                print('k = %d; λ = %.4g; η = %.4g; |fk| = %.4g; |Δx| = %.4g; tol = %.4g; ltol = %.4g.'
                      % (k, lam, self.linOp.eta, self.normfk, normdx, tol, self.ltol))

            # Solution found if condition here true
            if normdx <= tol:
                if self.debug: self.steps.append(deepcopy(dx))
                x += dx
                k += 1
                self.newton_success = True
                break

            if k > 0:
                w = dxbar - dx
                s = nnorm(w, xscale)
                mue = (normdxkm1 * normdxbar)/(s * normdx) * lam if s > 0 else np.inf
                lam = min(1.0, mue)

            # DO ADJUST DAMPENING STEP - the while loop below adjusts that
            reducted = False

            while True:
                if lam <= lambda_min:
                    # CONVERGENCE FAILURE!
                    warnings.warn('NLEQ-ERR: Failed to solve nonlinear system lambda0 = %.4g; (lambda = %.4g <= %.4g too small) after %d iterations! Current norm |dx| = %.4g > %.4g.'\
                                  % (lam0, lam, lambda_min, k, normdx, tol), RuntimeWarning)
                    self.newton_success = False
                    self.convergence_failure = True
                    self.lambda_limitation = True
                    break

                # New trial iterate
                xkp1 = x + lam * dx
                # if self.debug: print('\t\tdx:', dx, ' dx:', dx.coeffs)

                # Evaluate function at xkp1 - Don't regenerate the matrix!
                fxk = -self.linOp.rhs(xkp1)
                normfkp1 = self.to_state(fxk).norm()

                # Compute Newton correction -> uses residual as IC
                # Can also equivalently flip the sign of the RHS
                dxbar, success = self.isolve(fxk)

                if not success:
                    warnings.warn('NLEQ-ERR: Failed to solve linear system after %d iterations! Norm: %.6g' \
                                  % (k, normfkp1), RuntimeWarning)
                    self.newton_success = False
                    break

                # De-scale - don't need to scale here!
                normdxbar = nnorm(dxbar, xscale)

                # Compute the estimator
                theta = normdxbar / normdx
                s = 1.0 - lam
                w = dxbar - s * dx
                w_norm = nnorm(w, xscale)
                mue = (0.5 * normdx * lam * lam) / w_norm if w_norm > 0.0 else np.inf

                if self.debug:
                    print('\tk = %d; λ = %.4g; |fk|: %.4g; |Δx|: %.4g; |dx|: %.4g; Int[dx]: %.4g; |w|: %.4g, θ = %.4g; μ = %.4g; s = %.4g.'
                          % (k, lam, normfkp1, normdx, normdxbar, np.real(np.sum(np.sum(dx))), w_norm, theta, mue, s))

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
                        self.newton_success = True
                        break

                    qnerr_iter = (theta < THETA_MAX)
                else:
                    if lambda_new >= 4.0 * lam and not reducted:
                        lam = lambda_new

                        # Try and check again whether we are good now!
                        continue

                # If we get to the end of the loop we should quit the loop
                break

            # END OF DAMPENING ADJUSTMENT
            self.normfk = normfkp1

            # Need to check again since failures in the above while loop -> only break that loop!
            if not success:
                warnings.warn('NLEQ-ERR: Failed to solve linear system after %d iterations! Norm: %.6g' \
                                  % (k, self.normfk), RuntimeWarning)
                self.newton_success = False
                break

            # Save previous iterate for scaling purposes and accept new iterate
            if scale: w = x
            x = xkp1

            # Store steps for debug purposes
            if self.debug: self.iterates.append(deepcopy(x))
            if self.debug: self.steps.append(deepcopy(lam * dx))

            # callback
            success_callback = self.callback(k, lam * dx, lam * normdx, lam*normdx / normdx_prev)
            normdx_prev = lam * normdx
            if not success_callback:
                print('NLEQ-ERR: Failed callback indicates STOP after %d iterations!' % k)
                self.callback_failure = True
                self.newton_success = False
                break

            # NEXT STEP
            k += 1

            # Perform QNERR STEPS IF CHOSEN!
            if qnerr_iter:
                # if qnerr_iter failed we try to continue with NLEQ_ERR!
                x, success, kp, normdx = self.qnerr(LinOp, x, itr=k, dx0=dx, miter=miter-k+1,
                                                    nleqcalled=True, *args, **kwargs)
                k += kp

                if not success:
                    qnerr_iter = False

        # END MAIN SOLVER LOOP!
        return x, self.newton_success, k

    def qnerr(self, LinOp, x, dx0=None, itr=0, xscale=None, nleqcalled=False, miter=25, *args, **kwargs):
        """ Implements QNERR - ERRor-based Quasi-Newton algorithm

            Warning: The current implementation will not work stand-alone, but is designed to
            take input from NLEQ_ERR.
        """
        tol = self.ntol
        precond = kwargs.get('precond', self.req_precd)

        if not nleqcalled:
            self.neval = 0
            self.newton_success = False
            self.convergence_failure = False
            self.lambda_limitation = False
            self.solve_inner_failure = False
            self.callback_failure = False

        # Setup the iteration
        newton_success = False

        # allocate memory
        dx = np.empty(miter+2, dtype=object)
        sigma = np.zeros(miter+2)

        # Setup initial data
        k = 0
        normdx = 0
        skipstep = nleqcalled

        if not skipstep:
            # Do the first real newton step!
            self.lu = None
            self.qrchol = None
            self.linOp = LinOp(x)
            self.neval += 1
            self.precd = self.linOp.precond() if precond else None
            self.normfk = self.to_state(self.linOp.b).norm()

            # Call the solver
            dxk, success = self.isolve(-self.linOp.b)
            normdx = nnorm(dxk, xscale)
            #print('v%d = ' % k, np.asarray(dxk), ' |v| = %.4g' % dxk.norm())
            #print('dx%d= ' % k, dxk, ' |v| = %.4g' % dxk.norm())

            # We quit if we failed to solve the linear system.
            if not success:
                warnings.warn('QNERR: Failed to solve linear system after %d iterations! Norm: %.6g' \
                              % (k+1, self.normfk), RuntimeWarning)
                newton_success = False
                if not nleqcalled: self.newton_success = False
                return x, newton_success, k, normdx

            # descale - the newton step! Don't need to do this
            sigma[0] = normdx
            dx[0] = dxk

            # default value
            thetak = 0.5 * THETA_MAX

            success_callback = self.callback(k, dxk, normdx, thetak)
            if not success_callback:
                warnings.warn('QNERR: Failed callback indicates STOP after {0:d} iterations!'.format(k))
                self.callback_failure = True
                newton_success = False
                if not nleqcalled: self.newton_success = False
                return x, newton_success, k, normdx
        else:
            assert dx0 is not None, ''
            normdx = dx0.norm()
            sigma[0] = normdx**2
            dx[0] = dx0

        #print('x%d = ' % k, np.asarray(x))
        while not newton_success and k <= miter:
            if not skipstep:
                x += dx[k]
                #print('x%d = ' % k, np.asarray(x), ' ó = ', sigma[:k+1], '\n')
                #print('x%d = ' % k, ' ó = ', sigma[:k+1], ' ε = ', tol, '\n')
                if sigma[k] <= tol * tol:
                    k += 1
                    newton_success = True
                    if not nleqcalled: self.newton_success = True
                    break

            # no longer skipping steps
            skipstep = False
            fxk = -self.linOp.rhs(x)
            self.normfk = self.to_state(fxk).norm()
            v, success = self.isolve(fxk)
            #print('v%d = ' % (k+1), v, ' |v| = %.4g ' % nnorm(v))

            if not success:
                warnings.warn('QNERR: Failed to solve linear system after %d iterations!' % (k+1), RuntimeWarning)
                newton_success = False
                if not nleqcalled: self.newton_success = False
                break

            for i in range(1, k+1):
                alpha_bar = sprod(v, dx[i-1], xscale) / sigma[i-1]
                v += alpha_bar * dx[i]

            alpha_kp1 = sprod(v, dx[k], xscale) / sigma[k]
            thetak = sqrt(sprod(v, v, xscale) / sigma[k])
            #print('thetak = ', thetak, ' |v|2 = ', sprod(v, v, xscale))
            if self.debug:
                print('\tkq = %d; α = %.4g; θ = %.4g; |v|: %.4g; |Δx|: %.4g; ε = %.2g; σ = %.2g'
                      % (k, alpha_kp1, thetak, v.norm(), dx[k].norm(), tol, sigma[k]))

            # compute new step
            s = 1.0 - alpha_kp1
            # TODO: this sometimes causes states to reshape themselves!
            dx[k+1] = v
            dx[k+1] /= s
            #print('dx%d= ' % (k+1), np.asarray(dx[k+1]), ' |v| = %.4g' % dx[k+1].norm())
            sigma[k+1] = sprod(dx[k+1], dx[k+1], xscale)
            normdx = np.sqrt(sigma[k+1])
            #print('v%d = ' % (k+1), v, ' σ = ', sigma[k+1], ' |v| = %.4g ' % v.norm(), ' Θ = ', thetak)
            k += 1

            # callback
            success = self.callback(itr + k, dx[k], normdx, thetak)

            if thetak > THETA_MAX:
                if not nleqcalled:
                    warnings.warn('QNERR: Failed to solve nonlinear system after {0:d} iterations! θ = {1:.4g} greater than 1/2 tolerance.'
                              .format(k+1, thetak), RuntimeWarning)
                newton_success = False
                self.lambda_limitation = True
                if not nleqcalled: self.newton_success = False
                break

        # END MAIN SOLVER LOOP!
        return x, newton_success, k, normdx
