#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
from math import sqrt
from copy import deepcopy

import numpy as np

from newton.newton_base import NewtonBase, NewtonErrors
from newton.qnerr import qnerr
from newton.nleq_err import nleq_err
from newton.random import RandomVector

from newton.sys.MoorePenroseSystem import MoorePenroseSystem
from newton.sys.FoldContSystem import NewtonGaussFoldSystem

from states.mp_state import MoorePenroseState

from bif.point import Point
from bif.monitor import SparseMonitor
from bif.tools import get_coords, get_offset, BranchType


class NewtonGaussPredictorCorrector(NewtonBase):
    """

        Class implementing Newton-Gauss continuation making use of the
        linear operators Schur complement.

    """

    def __init__(self, system, *args, **kwargs):

        # The possible modes:
        linSys = {BranchType.ONE_PAR: MoorePenroseSystem,
                  BranchType.TWO_PAR: NewtonGaussFoldSystem}

        self.method = kwargs.get('method', BranchType.ONE_PAR)
        linOp = linSys.get(self.method, MoorePenroseSystem)

        # Call super constructor
        super().__init__(system, linOp, *args, **kwargs)

        # The core newton method
        #self.method = kwargs.pop('method', 'qnerr')
        self.method = 'qnerr'
        self.newton = qnerr if self.method == 'qnerr' else nleq_err

        # Theoretical step size estimates
        self.theta_max = kwargs.pop('theta_max', 0.5 if self.method == 'qnerr' else 1.0)
        self.theta_bar = kwargs.pop('theta_bar', 0.25)
        self.theta_thr = kwargs.pop('theta_thr', 0.1)
        self.safety_factor = kwargs.pop('safety_factor', 0.5)

        # Create the store for the random vector
        self.rvec = RandomVector(self.system.n_disc, system.neqn, domain=system.domain)

        # Reset solver? FIXME?
        self.reset()

    @property
    def correct_stepsize(self):
        # Corrector only works when a step-size is known!
        # This is only called when an iteration has failed!
        # Thus this should always be less than 1
        return min(1.0, sqrt(self.safety_factor * self.theta_bar / max(self.theta_thr, self.thetak)))

    @property
    def predict_stepsize(self):
        # If we failed we can't compute anything!
        if self.iter == 0:
            return 1. / sqrt(self.theta_thr)

        thr = self.safety_factor * self.theta_bar / max(self.theta_thr, self.thetak)
        return sqrt(thr * self.norm_dx0 / (self.pred_error * self.c0))

    def reset(self):
        # Newton-Gauss fails for anything else -> for force this!
        self.theta     = 1.0
        self.alpha     = 1.0

        # Define these monitors
        self.norm_dx0  = 1
        self.c0        = 1
        self.theta0    = self.theta_bar
        self.thetai    = 1
        self.thetak    = 2 * self.theta_bar
        self.max_theta = 0
        self.iter      = 0

        # The predicted solution
        self.pred_pt = None

    def init(self, state0):
        ndir = np.ones_like(state0)
        LinOp = self.linop(state0, self.system, ndir, self.rvec, dtype=self.dtype)

        # Compute the bif points
        res = LinOp.rhs(state0)
        dx, success = LinOp.solve(res)
        tau = LinOp.bif_monitor(state0, self.rvec)
        ndir = LinOp.tangent()

        # Create the monitor
        return ndir, SparseMonitor(ndir.a, tau)

    def tangent(self, un, tangent):
        # Compute the operator at the newly computed point
        LinOp = self.linop(un, self.system, tangent, None, dtype=self.dtype)

        # Compute the new tangent
        return LinOp.tangent(), LinOp.left_nullvector()

    def callback(self, system, itr, dx, normdx, thetak):
        self.thetak = thetak

        if itr == 0:
            """ Step size control statistics:

                1) Compute angle between predictor tangent and tangent at first prediction point
                2) Stores the norm of the first correction
                3) Computes the first contraction rate.

            """
            self.norm_dx0 = normdx

            # This is called after the first QNERR step -> QR at guess is computed already!
            tangent_pred = system.tangent()
            self.c0 = abs(np.dot(tangent_pred, self.opt.tangent))

        elif itr == 1:
            self.theta0 = dx.norm() / normdx
            self.thetai = self.theta0
            self.max_theta = thetak
            # return self.theta0 <= self.theta_max
        else:
            self.theta0 = dx.norm() / normdx
            self.max_theta = max(self.max_theta, thetak)

        # By default if we don't have enough information continue!
        return thetak <= self.theta_max or itr == 0

    def solve(self, pt, ds, miter=25, predictor=True, check_angle=True,
              angle_tolerance=0.1, compute_monitor=True, *args, **kwargs):

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
        # save the origin point
        self.opt = deepcopy(pt)
        state = self.opt.state

        # reset
        self.reset()
        success = False

        # Set shape -> important to how to interpret coefficient vectors!
        self.cpar = state.cpar

        ###############################
        # Step 1: Predictor           #
        ###############################
        if predictor:
            s1 = state + ds * pt.tangent
        else:
            s1 = deepcopy(state)

        # Set shapes after simplification of initial condition!
        self.function_type = 'trig' if state.u.istrig else 'cheb'

        # copy this for the moment for step adjustment: TODO: do we need to copy?
        self.pred_pt = deepcopy(s1)

        ###############################
        # Step 2: Solve Corrector     #
        ###############################
        tol = kwargs.pop('ntol', 1e-8)

        # The operator computing the Frechet derivative
        def LinOp(u):
            return self.linop(u, self.system, self.opt.tangent, self.rvec, dtype=self.dtype)

        un, success, iters, normdx, neval, system = qnerr(
            LinOp, s1, tol, miter=miter, adapt=True, callback=self.callback,
            inner=np.dot, theta_max=self.theta_max, verb=False,
            *args, **kwargs)

        # Compute the prediction error
        self.pred_error = (un - self.pred_pt).norm()

        # If not success return
        if not success:
            self.status = NewtonErrors.NonlinSolFail
            return None, success, iters

        ###################################################
        # Step 3: Finish up!                              #
        ###################################################
        ndir = system.tangent()

        # Check the change in the parameterized parameter!
        bdir = system.Faa
        chi = system.internal_embedding_parameter

        # Compute the position: [ROW, COL]
        # This is required since the old point underwent truncation so it may be shorter than the
        # currently used discretization!
        loc = get_coords(chi, un.shape)
        chi_old = get_offset(loc, self.opt.state.shape) if chi != -1 else -1

        # Now compute the deviation of the continuation parameter
        da = np.asarray(un)[chi] - np.asarray(self.opt.state)[chi_old]

        # Compute the operator at the newly computed point
        ## if compute_monitor:
        ##     LinOp = self.linop(un, self.system, ndir, self.rvec, dtype=self.dtype)

        ##     # Compute the new tangent
        ##     # ndir = LinOp.tangent()

        ##     # Fold monitor
        ##     # Since we just recalculated the tangent its parameter is proportional to
        ##     # the determinant of Fx; see MoorePenroseSystem comments.
        ##     tau = LinOp.bif_monitor()

        ##     if tau * self.opt.monitor.tau < 0:
        ##         bdir = LinOp.bif_dir()
        tau = 1.0

        # Correct tangent sign errors due to changing natural orientation around turning points.
        if chi != -1 and np.asarray(ndir)[chi] * da < 0:
            ndir = ndir.flip()

        # Check that the angle between consecutive tangent vectors is small enough!
        self.alpha = np.dot(ndir, self.opt.tangent)
        if check_angle and not self.alpha >= angle_tolerance:
            self.status = NewtonErrors.ContAngleFail
            self.thetak = self.theta_max
            return None, False, iters

        # Create the monitor object
        nmon = SparseMonitor(ndir.a, tau) if compute_monitor else None

        # set status
        self.status = NewtonErrors.Success

        # Create the new point if everything was successful
        npt = Point(state=un, tangent=ndir, chi=chi, monitor=nmon, bdir=bdir)

        # Finally return everything
        return npt, success, iters

    def solve2(self, pt, ds, miter=100, phi=None, psi=None,
               op_kwargs={}, check_angle=True, compute_monitor=True, *args, **kwargs):

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
        # save the origin point
        self.opt = deepcopy(pt)

        # reset
        self.reset()
        success = False
        angle_tolerance = kwargs.pop('angle_tolerance', 0.3)

        ###############################
        # Step 1: Predictor           #
        ###############################
        print('pt = ', pt)
        print('state = ', pt.state)
        print('tang  = ', pt.tangent)

        s1 = pt.state + ds * pt.tangent if (ds != 0.0) else deepcopy(pt.state)
        s1 = MoorePenroseState(u=s1, v=pt.tangent)

        # Create a point of the predicted point
        # self.pred_pt = Point(s1, tangent=pt.tangent)

        # Set shapes after simplification of initial condition!
        self.function_type = 'trig' if pt.state.u.istrig else 'cheb'

        # copy this for the moment for step adjustment: TODO: do we need to copy?
        self.pred_pt = deepcopy(s1)

        ###############################
        # Step 2: Solve Corrector     #
        ###############################
        tol = kwargs.pop('ntol', 1e-5)  # This still causes failures if chosen too tight.

        # The operator computing the Frechet derivative
        def LinOp(un, *args, **kwargs):
            return self.linop(un, self.system, phi, psi, **op_kwargs, dtype=self.dtype)

        un, success, iters, normdx, neval, system = nleq_err(
            LinOp, s1, tol=tol, miter=miter, callback=self.callback, use_qnerr=False,
            inner=np.dot, theta_max=self.theta_max, debug=True, nonlin='extremely',
            *args, **kwargs)

        #un, success, iters, normdx, neval, system = qnerr(
        #    LinOp, s1, tol=tol, miter=miter, callback=self.callback,
        #    inner=np.dot, theta_max=self.theta_max, debug=True,
        #    *args, **kwargs)

        # Compute the prediction error
        self.pred_error = (un - self.pred_pt).norm()

        # If not success return
        if not success:
            self.status = NewtonErrors.NonlinSolFail
            return None, success, iters, None

        ###################################################
        # Step 3: Finish up!                              #
        ###################################################
        # ndir = system.tangent()

        # Check the change in the parameterized parameter!
        bdir = None

        ### chi = system.internal_embedding_parameter
        ### # Compute the position: [ROW, COL]
        ### # This is required since the old point underwent truncation so it may be shorter than the
        ### # currently used discretization!
        ### loc = get_coords(chi, un.shape)
        ### chi_old = get_offset(loc, self.opt.state.shape) if chi != -1 else -1

        ### # Now compute the deviation of the continuation parameter
        ### da = np.asarray(un)[chi] - np.asarray(self.opt.state)[chi_old]

        # Compute the operator at the newly computed point
        if compute_monitor:
            LinOp = self.linop(un, self.system, ndir, self.rvec, dtype=self.dtype)

            # Compute the new tangent
            ndir = LinOp.tangent()

            # Fold monitor
            # Since we just recalculated the tangent its parameter is proportional to
            # the determinant of Fx; see MoorePenroseSystem comments.
            tau = LinOp.bif_monitor()

            if tau * self.opt.monitor.tau < 0:
                bdir = LinOp.bif_dir()

        # Correct tangent sign errors due to changing natural orientation around turning points.
        ### if chi != -1 and np.asarray(ndir)[chi] * da < 0:
        ###     ndir = ndir.flip()

        # Check that the angle between consecutive tangent vectors is small enough!
        #self.alpha = np.dot(ndir, self.opt.tangent)
        self.alpha = 1
        if check_angle and not self.alpha >= angle_tolerance:
            self.status = NewtonErrors.ContAngleFail
            self.thetak = self.theta_max
            return None, False, iters, None

        # Create the monitor object
        nmon = SparseMonitor(ndir.a, tau) if compute_monitor else None

        # set status
        self.status = NewtonErrors.Success

        print('tangent = ', un.v)
        un.v = un.v.normalize()

        # Create the new point if everything was successful
        npt = Point(state=un, tangent=un.v, monitor=nmon, phi=system.v, bdir=system.w)

        # Finally return everything
        return npt, success, iters, system
