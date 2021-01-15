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
from sparse.csr import eliminate_zeros_csr

from funpy.fun import minandmax
from funpy.fun import Fun, h1norm, norm, norm2
from funpy.cheb.chebpts import quadwts
from funpy.cheb.diff import computeDerCoeffs
from funpy.states.State import ContinuationState
from funpy.states.tp_state import TwoParameterState
from funpy.states.parameter import Parameter
from funpy.newton.pseudo_arclength import PseudoArcContinuationCorrector
from funpy.newton.newton_gauss import NewtonGaussContinuationCorrector
from funpy.newton.newton import NewtonBase, sprod
from funpy.linalg.qr_solve import QRCholesky
from nlep.nullspace import right_null_vector
from funpy.newton.deflated_residual import DeflatedResidual
from funpy.support.tools import orientation_y, logdet, functional, Determinant

from bif.operator import Operator
from bif.point import Point, Monitor
from bif.fold_system import AugmentedFoldSystem

# NUMBERS FOR NLEQ_ERR
THETA_MAX = 0.5
THETA_BAR = 0.25  # Theoretically this is 0.25; but practice shows this needs to be tighter


class NewtonGaussPredictorCorrector(NewtonBase):
    def __init__(self, system, system_type='ng', *args, **kwargs):
        # set this
        self.to_state = self.to_state_cont if system_type == 'ng' else self.to_state_tp

        system_type = dict(ng=NewtonGaussContinuationCorrector,
                          fold=AugmentedFoldSystem).get(system_type, system_type)
        super().__init__(system, system_type, *args, **kwargs)
        self.reset()

    def reset(self):
        # Newton-Gauss fails for anything else -> for force this!
        self.theta = 1.0

        # Define these monitors
        self.norm_dx0 = 1
        self.c0 = 1
        self.thetak = 2 * THETA_BAR

        # The predicted solution
        self.pred_pt = None

    def to_state_cont(self, coeffs, ishappy=True):
        """ Constructs a new state from coefficients """
        m = coeffs.size // self.n_eqn
        if m * self.n_eqn == coeffs.size:
            return self.to_fun(coeffs, ishappy=ishappy)

        soln = self.to_fun(coeffs[:-1], ishappy=ishappy)
        ap = Parameter(**{self.cpar: np.real(coeffs[-1].item())})
        state = ContinuationState(a=ap, u=np.real(soln)) # , n=max(1, soln.shape[0]))
        return state

    def to_state_tp(self, coeffs, ishappy=True):
        """ Constructs a new state from coefficients """
        m = (coeffs.size - 2) // self.n_eqn

        if m * self.n_eqn < coeffs.size - 2:
            m = (coeffs.size - 1) // self.n_eqn
            soln = self.to_fun(coeffs[:m], ishappy=ishappy)
            ap = Parameter(**{self.cpar: np.real(coeffs[-1].item())})
            phi = self.to_fun(coeffs[m:-1], ishappy=ishappy)
            pp = Parameter(**{self.bpar: 0.0})
            return TwoParameterState(p1=pp, p2=ap, u=np.real(soln), phi=np.real(phi))

        soln = self.to_fun(coeffs[:m], ishappy=ishappy)
        ap = Parameter(**{self.cpar: np.real(coeffs[-1].item())})
        phi = self.to_fun(coeffs[m:-2], ishappy=ishappy)
        pp = Parameter(**{self.bpar: np.real(coeffs[-2].item())})
        return TwoParameterState(p1=pp, p2=ap, u=np.real(soln), phi=np.real(phi))

    def to_fun(self, coeffs, ishappy=True):
        """ Constructs a new state from coefficients """
        m = coeffs.size // self.n_eqn
        soln = Fun(coeffs=coeffs.reshape((m, self.n_eqn), order='F'), simplify=False,
                   domain=self.system.domain, ishappy=ishappy, type=self.function_type)
        return soln

    def solve_predictor(self, state, ds, direction, *args, **kwargs):
        """ Computes the predictor. Here we use a simple Euler step """
        return state + ds * direction

    def solve_corrector(self, x, miter=30, restricted=True, scale=False,
                        method='nleq', *args, **kwargs):
        """ The main entry point to the nonlinear solver for Newton Gauss continuation.

            TODO: finish me!
        """
        self.qrchol = None

        # Create a mapping that maps u -> D_u F
        ks = kwargs.get('ks', [])

        # The operator computing the Frechet derivative
        LinOp = lambda u: self.linop(u, self.system, ks=ks, dtype=self.dtype)

        if method == 'nleq':
            un, success, iters = self.nleq_err(LinOp, x, miter=miter, restricted=restricted,
                                               scale=scale, *args, **kwargs)
        elif method == 'qnerr':
            un, success, iters, normdx = self.qnerr(LinOp, x, miter=miter, restricted=restricted,
                                                    scale=scale, *args, **kwargs)
        else:
            raise RuntimeError("Unknown nonlinear solver method %s." % method)

        return un, success, iters

    def get_matrix(self, state, *args, **kwargs):
        """
        Get the matrix of the linearization. Note that this matrix cannot be used for solving
        the system, since it will not have the rows required for the reduction of boundary
        conditions.
        """
        # Continue with the Sparse QR
        LinOp = self.linop(state, self.system,
                           dtype=np.complex if state.u.istrig else np.float, **kwargs)

        # Get the basis inverse
        op  = LinOp.matrix_full(state)
        adj = LinOp.adjoint(state)

        # Assemble an operator
        op = Operator(op, adjoint=adj, b=LinOp.b)
        return op

    def compute_qr(self, state, *args, **kwargs):
        precond = kwargs.get('precond', self.req_precd)
        assert state is not None, 'Recomputing requires a state!'
        LinOp = self.linop(state, self.system,
                           dtype=np.complex if state.u.istrig else np.float, **kwargs)

        # This is P-inverse; where P = diag(A)
        self.precd = LinOp.precond() if precond else None
        self.B = LinOp.to_matrix()
        self.P = self.precd.to_matrix()
        self.A = (self.B * self.P).todense()

        # Compute the QR-decomposition
        eps = 1e-14
        self.qrchol = QRCholesky(self.A, eps=eps, rank=state.rank, *args, **kwargs)

    def solve_direction_svd(self, state, compute_qr=True, *args, **kwargs):
        return self.solve_direction_qr(state=state, compute_qr=compute_qr, *args, **kwargs)

    def solve_direction_qr(self, state=None, compute_qr=True, *args, **kwargs):
        if compute_qr:
            self.cpar = state.cpar
            self.bpar = state.bname
            self.n_eqn = state.shape[1]
            self.compute_qr(state)

        # Use the QR decomposition to solve the problem now
        coeffs = self.qrchol.tangent_vector()
        coeffs = self.P * coeffs

        # Assemble the direction
        direction = self.to_state(coeffs)
        direction = direction.normalize()
        return direction

    def get_monitor(self, state=None, compute_qr=False):
        if compute_qr:
            self.compute_qr(state)

        # Compute monitor (d_chi, d_lambda)
        d_chi = self.qrchol.det()
        d_lam = self.qrchol.detH()
        return Monitor(d_chi, d_lam)

    @property
    def smin(self):
        pass

    @property
    def smax(self):
        pass

    @property
    def correct_stepsize(self):
        # Corrector only works when a step-size is known!
        # This is only called when an iteration has failed!
        # Thus this should always be less than 1
        # TODO: why is this not always the case!!!
        return min(1.0, sqrt(THETA_BAR / self.thetak))

    @property
    def predict_stepsize(self):
        # If we failed we can't compute anything!
        theta0 = max(0.01, np.max(self.thetas))
        factor = sqrt(self.norm_dx0 * THETA_BAR / (self.pred_error * theta0 * abs(self.c0)))
        return factor

    def callback(self, itr, dx, normdx, thetak):
        self.thetas[max(0, itr-1)] = thetak
        self.thetak = thetak

        if itr == 0:
            """ Step size control statistics:

                1) Compute angle between predictor tangent and tangent at first prediction point
                2) Stores the norm of the first correction
                3) Computes the first contraction rate.

            """
            self.norm_dx0 = normdx

            # This is called after the first QNERR step -> QR at guess is computed already!
            tangent_pred = self.solve_direction_qr(compute_qr=False)
            self.c0 = sprod(tangent_pred, self.opt.tangent)

        # By default if we don't have enough information continue!
        return True

    def solve(self, pt, ds, miter=25, method='qnerr', predictor=True, *args, **kwargs):

        """ This function implements the core of the basic newton method

        Inputs: state - must be convertible to np.ndarray
                tol   - tolerance of the Newton iteration
                      - negative -> parameter must initially decrease.

                preserve_orientation - Checks that the input direction and output direction have
                the same orientation. This is to avoid flips which seems to commonly occur when
                dealing with Fredholm operators.

        Notes: 1) Curiously restarting the iteration at a higher discretization size with the
        previous solution leads to linear system convergence failures in roughly 10% of cases.
        """
        success = False
        # save the origin point
        self.opt = deepcopy(pt)

        state = self.opt.state
        n = state.n
        m = state.m

        # reset
        self.reset()

        # contraction ratio
        self.thetas = np.zeros(miter+2)
        self.thetas[0] = 0.5 * THETA_MAX
        self.rank = 0

        # set shape
        self.shape = (n, m)
        self.function_type = 'trig' if state.u.istrig else 'cheb'

        # Set shape -> important to how to interpret coefficient vectors!
        self.n_eqn = state.shape[1]
        self.cpar = state.cpar
        self.ntol = kwargs.get('tol', self.ntol)
        self.debug = kwargs.get('debug', self.debug)
        self.verb = kwargs.get('verb', self.verb)

        # Reset the stored outer_v values
        self.cleanup()

        ###############################
        # Step 1: Predictor           #
        ###############################
        if predictor:
            s1 = self.solve_predictor(state, ds, pt.tangent, *args, **kwargs)
        else:
            s1 = deepcopy(state)

        # copy this for the moment for step adjustment: TODO: do we need to copy?
        self.pred_pt = deepcopy(s1)

        ###############################
        # Step 2: Try Corrector       #
        ###############################
        un, success, its = self.solve_corrector(s1, miter=miter, method=method, *args, **kwargs)

        # Compute the prediction error
        self.pred_error = (un - self.pred_pt).norm()

        ###################################################
        # Step 3: Compute new direction for the next step #
        ###################################################
        new_dir = self.solve_direction_qr(un, compute_qr=True, *args, **kwargs)

        # flip around if the direction is wrong!
        angle = sprod(new_dir, self.opt.tangent)
        if angle < 0:
            new_dir = new_dir.flip()

        # Check: do I have to do this work?
        angle = sprod(new_dir, self.opt.tangent)
        if not angle >= 0.3:
            return None, False, its

        # Compute monitor (d_chi, d_lambda)
        nmon = self.get_monitor()
        npt = Point(state=un, tangent=new_dir, monitor=nmon,
                    chi=self.qrchol.internal_embedding_parameter)

        # Finally return everything
        return npt, success, its
