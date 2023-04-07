import numpy as np
import scipy.linalg as LA
import scipy.sparse as sps
from math import sqrt
from copy import deepcopy

from fun import minandmax, norm
from cheb.chebpts import quadwts
from states.cont_state import ContinuationState
from newton.pseudo_arclength import PseudoArcContinuationCorrector
from newton.newton_base import NewtonBase
from linalg.nullspace import right_null_vector
from support.tools import orientation_y, logdet, functional

THETA_MAX = 0.5
THETA_BAR = 0.25  # Theoretically this is 0.25; but practice shows this needs to be tighter


class NewtonPredictorCorrector(NewtonBase):
    def __init__(self, system, *args, **kwargs):
        super().__init__(system, PseudoArcContinuationCorrector, *args, **kwargs)
        self.dorder = kwargs.get('dorder', 0)
        self.theta = kwargs.get('theta', 0.5)

        # stats
        self.contraction = -1.0
        self.norm_dx0    = -1.0
        self.angle       = -1.0

        # nominal values
        self.nom_curvature = kwargs.pop('curvature', 1.0)
        self.nom_distance  = kwargs.pop('distance', 1.0)
        self.nom_angle     = kwargs.pop('angle', 2.0)

        # stats - of Newton methods
        self.contraction = THETA_BAR
        self.norm_dx0    = -1.0
        self.angle       = -1.0
        self.c0          = -1.0
        self.prederror   = np.inf

    def to_state(self, coeffs, ishappy=True):
        """ Constructs a new state from coefficients """
        soln = self.to_fun(coeffs[:-1], ishappy=ishappy)
        state = ContinuationState(a=np.real(coeffs[-1]), u=np.real(soln),
                                  cpar=self.cpar, n=max(1, soln.shape[0]), theta=self.theta)
        return state

    def solve_predictor(self, state, ds, direction, *args, **kwargs):
        """ Computes the predictor. Here we use a simple Euler step """
        return state + ds * direction

    def solve_corrector(self, x, state, ds, miter=30, direction=None, restricted=True,
                        scale=False, theta=0.5, method='nleq', adapt=True, *args, **kwargs):
        """ The main entry point to the nonlinear solver for pseudo arclength continuation.

            Note that NLEQ_ERR + QNERR has the property of solutions being attracted to two kinds
            of points: 1) Solutions, and 2) Singular points i.e. points where the Frechet
            derivative is singular.

            Bifurcation points during pseudo-arclength continuation are an example of such singular
            points. For this reason, we allow the use of standard newton steps during
            pseudo-arclength continuation.

            That is if we detect NLEQ_ERR failure due to a singular matrix, usually indicated by λ
            that approaches zero.

            Yet another idea: When encountering trouble in NLEQ -> use a few standard newton steps
            -> then apply NLEQ

            Arguments:

                x: Member of the space (a, u) where u in X some function space and a in R

        """
        self.lu = None
        self.qrchol = None
        self.theta = theta

        # Create a mapping that maps u -> D_u F
        ks = kwargs.get('ks', [])

        # The operator computing the Frechet derivative
        LinOp = lambda u: self.linop(u, state, self.system, direction=direction, dorder=self.dorder,
                                     ds=ds, ks=ks, theta=self.theta, dtype=self.dtype)

        while True:
            if method == 'nleq':
                un, success, iters = self.nleq_err(LinOp, x, miter=miter, restricted=restricted,
                                                   scale=scale, *args, **kwargs)
            elif method == 'qnerr':
                un, success, iters, normdx = self.qnerr(LinOp, x, miter=miter, restricted=restricted,
                                                        scale=scale, *args, **kwargs)
            elif method == 'classic':
                un, success, iters = self.newton_step(LinOp, x, miter=miter, *args, **kwargs)

                # We only succeeded if the final state is "close" to the initial state.
                success &= abs(x.a - un.a) < 1.25 * abs(ds)
            else:
                raise RuntimeError("Unknown nonlinear solver method %s." % method)


            # Loop control!
            break

            #if self.lambda_limitation and abs(ds) <= 1e-3 and count < 1:
            #    # Reduce rank by one!
            #    self.rank -= 1
            #    count+=1
            #else:
            #    break

        return un, success, iters

    def get_matrix(self, state, *args, **kwargs):
        # Continue with the Sparse QR
        theta = kwargs.pop('theta', 0.5)
        LinOp = self.linop(state, state, self.system, theta=theta, dorder=self.dorder,
                           dtype=np.complex if state.u.istrig else np.float, **kwargs)
        return LinOp.to_matrix().transpose().tocsc()

    def solve_direction_sparse(self, state, m=1, *args, **kwargs):
        self.function_type = 'trig' if state.u.istrig else 'cheb'

        # If we have a complex matrix -> can't use suitesparse at the moment
        # TODO: Is this a python interface issue?
        if state.u.istrig:
            return self.solve_direction_qr(state, *args, **kwargs)

        # Continue with the Sparse QR
        import sparseqr
        self.cpar = state.cpar
        self.n_eqn = state.shape[1]
        theta = kwargs.get('theta', 0.5)

        LinOp = self.linop(state, state, self.system, dorder=self.dorder,
                           dtype=np.complex if state.u.istrig else np.float, **kwargs)

        A = LinOp.to_matrix().transpose().tocsc()
        n = LinOp.shape[0]
        Q, R, E, rank = sparseqr.qr(A)

        # Check that rank is only deficient by one!
        if rank < n - 1:
            raise RuntimeError('Cannot compute new direction vector at a singular point! Matrix has rank {0:d} and size {1:d}'.format(rank, n))

        null_basis = np.asarray(Q.tocsc()[:, -1].todense())
        direction = self.to_state(null_basis.flatten())
        direction.theta = theta
        # Normalize the direction!
        direction = direction.normalize()

        orientation = kwargs.get('orientation', None)
        if orientation is not None:
            new_orient = orientation_y(direction.u)
            if new_orient * orientation < 0:
                direction *= -1.0

        #print('direction solved = ', direction.u.coeffs.T)
        return direction

    def solve_direction_qr(self, state, *args, **kwargs):
        self.cpar = state.cpar
        self.n_eqn = state.shape[1]
        theta = kwargs.pop('theta', 0.5)
        self.function_type = 'trig' if state.u.istrig else 'cheb'
        LinOp = self.linop(state, state, self.system, theta=theta, dorder=self.dorder,
                           dtype=np.complex if state.u.istrig else np.float, **kwargs)

        # Create the required dense matrices
        A = LinOp.to_matrix().transpose().todense()
        # n = LinOp.shape[0]
        Q, R, P = LA.qr(A, overwrite_a=True, mode='full', pivoting=True)

        # Check that rank is only deficient by one!
        #if rank < n - 1:
        #    raise RuntimeError('Cannot compute new direction vector at a singular point! Matrix has rank {0:d} and size {1:d}'.format(rank, n))

        null_basis = np.asarray(Q[:, -1])
        direction = self.to_state(null_basis.flatten())
        direction.theta = theta
        # Let's only take the real part of the function -> TODO: why does the above QR give me such
        # large imaginary components???
        direction = np.real(direction)

        # Normalize the direction!
        direction = direction.normalize()

        orientation = kwargs.pop('orientation', None)
        if orientation is not None:
            new_orient = orientation_y(direction.u)
            if new_orient * orientation < 0:
                direction *= -1.0

        return direction

    def solve_direction_svd(self, state, *args, **kwargs):
        self.cpar = state.cpar
        self.n_eqn = state.shape[1]
        theta = kwargs.pop('theta', 0.5)
        self.function_type = 'trig' if state.u.istrig else 'cheb'
        LinOp = self.linop(state, state, self.system, theta=theta, dorder=self.dorder,
                           dtype=np.complex if state.u.istrig else np.float, **kwargs)

        # Create the required dense matrices
        A = LinOp.to_matrix()
        Ad = A.todense()
        u, s, v = LA.svd(Ad, overwrite_a=True)
        sing = np.argmin(s)

        # left_null = u[:, sing]  # Null vector of transpose
        righ_null = v[sing, :]  # Null vector of the matrix

        # Map to proper space representation
        direction = self.to_state(righ_null.flatten())
        direction.theta = theta

        # Let's only take the real part of the function -> TODO: why does the above QR give me such
        # large imaginary components???
        direction = np.real(direction)

        # Normalize the direction!
        direction = direction.normalize()

        # Compute sign of the direction
        sign, logdet = self.compute_orientation(state=state, direction=direction)
        direction.sign = sign
        return direction

    def solve_nullvector_svd(self, state, *args, **kwargs):
        self.cpar = state.cpar
        self.n_eqn = state.shape[1]
        theta = kwargs.pop('theta', 0.5)
        self.function_type = 'trig' if state.u.istrig else 'cheb'
        LinOp = self.linop(state, state, self.system, theta=theta, dorder=self.dorder,
                           dtype=np.complex if state.u.istrig else np.float, **kwargs)

        # Create the required dense matrices
        A = LinOp.Du.to_matrix().todense()
        u, s, v = LA.svd(A, overwrite_a=True)
        sing = np.argmin(s)

        left_null = u[:, sing]  # Null vector of transpose
        righ_null = v[sing, :]  # Null vector of the matrix

        # Map to proper space representation
        right_null_fun = self.to_fun(righ_null.flatten())

        # Normalize the direction!
        right_null_fun /= norm(right_null_fun)

        # Map to proper space representation
        left_null_fun = self.to_fun(left_null.flatten())

        # Normalize the direction!
        inner_prod = sprod(left_null_fun, right_null_fun)
        left_null_fun /= inner_prod
        return right_null_fun, left_null_fun

    def solve_direction(self, state, *args, **kwargs):
        self.cpar = state.cpar
        self.n_eqn = state.shape[1]
        self.function_type = 'trig' if state.u.istrig else 'cheb'
        theta = kwargs.pop('theta', 0.5)
        LinOp = self.linop(state, state, self.system, theta=theta, **kwargs)
        coeffs = right_null_vector(LinOp.todense(), normalize=False)
        direction = self.to_state(coeffs)
        direction.theta = theta
        # Normalize the direction!
        return direction.normalize()

    def compute_orientation(self, L0=None, state=None, direction=None, *args, **kwargs):
        """ The orientation is defined as the determinant of the Jacobian i.e.

            Orientation(u(s), dot(u)(s)) = det ()
        """
        if L0 is None:
            assert state is not None and direction is not None, ''
            theta = direction.theta
            L0 = self.linop(state, state, self.system, theta=theta, direction=direction,
                            dorder=self.dorder,
                            dtype=np.complex if state.u.istrig else np.float, **kwargs)
        else:
            L0 = self.linOp

        # Get the matrix
        F0 = L0.to_matrix()
        sign, ldet = logdet(F0)
        return sign, ldet

    def swibra(self, state, direction, *args, **kwargs):
        phi, psi = self.solve_nullvector_svd(state, *args, **kwargs)
        w = np.tile(quadwts(state.n), (state.m,))

        phi = phi.prolong(state.n)
        psi = psi.prolong(state.n)
        theta = state.theta

        alpha0 = direction.a
        alpha1 = sprod(psi, direction.u)
        phi0 = (direction.u - alpha1 * phi) / alpha0
        phi0 = phi0.prolong(state.n)

        # Compute current right hand side
        L0 = self.linop(state, state, self.system, theta=theta, dorder=self.dorder,
                           dtype=np.complex if state.u.istrig else np.float, **kwargs)
        F0 = L0.to_matrix().todense()

        # perturb the state
        delta = kwargs.pop('delta', 0.01)
        un = state.u + delta * phi
        sd = ContinuationState(a=state.a, u=un, cpar=self.cpar, n=state.n, theta=state.theta)
        L0 = self.linop(sd, sd, self.system, theta=theta, dorder=self.dorder,
                           dtype=np.complex if state.u.istrig else np.float, **kwargs)
        F1 = L0.to_matrix().todense()
        dF = F1 - F0

        # compute the approximations
        n = np.product(state.shape)

        # Compute the functionals
        psic = functional(psi).flatten(order='F').reshape((1, n))
        print('psic = ', psic.shape)
        phic = phi.coeffs.flatten(order='F').reshape((n, 1))
        phi0c = phi0.coeffs.flatten(order='F').reshape((n, 1))

        # FIXME fix all the basis issues here!!!!
        Sinv = L0.Du.colloc.iconvert(0)
        SS = np.empty((2,2), dtype=object)
        SS[0,0]=Sinv
        SS[1,1]=Sinv
        S = sps.bmat(SS)
        a1 = (np.dot(psic, S * np.dot(dF[:-1, :-1], phic)) / delta).item()
        b1 = (np.dot(psic, S * np.dot(dF[:-1, :-1], phi0c) + dF[:-1, -1]) / delta).item()

        alpha1bar = - (a1 * alpha1 / alpha0 + 2 * b1)
        un = alpha1bar * phi + a1 * phi0
        ndirc = ContinuationState(a=a1, u=un, cpar=self.cpar, n=state.n, theta=state.theta)
        ndirc = ndirc.normalize()

        sign, ldet = self.compute_orientation(state=state, direction=ndirc)
        ndirc.sign = sign
        return ndirc

    def estimate_stepsize(self, state, direction, ds, factor=4, ds_max=0.15, *args, **kwargs):
        """ Estimates the "best" step size ds to use. Here best is not optimal in any mathematical
        sense but a seemingly good estimate. Currently, mainly tested for mesa-pattern solutions.

        Given a solution (a_{0}, u_{0}) and the derivative with respect to the arc-length by
        (\dot{a}, \dot{u}). Then the Euler step is given by:

            u_{new} = u_{0} + Δs \dot{u}
            a_{new} = a_{0} + Δs \dot{a}

        The idea of this estimate is to scale Δs such that the shape of u_{new} is not too
        different from u_{0}. For mesa-patterns the following works well.

                   A_u
        Δs =  --------------
               k A_{\dot{u}}

        where A_{f} = Total variation of the function (here we compute that as the difference
        between max and min). Note that since we expect the functions to be differentiable the true
        method would be to compute Int[|f'(x)|] but for that we require a correct splitting of
        domains due to the absolute value. This needs to be fixed in the function backend.

        For mesa-patterns k = 4 appears to be a good choice.

        """
        state_u = state.u
        dirc_u = direction.u

        # When the functions are constant just return requested ds
        if state_u.shape[0] == 1 or dirc_u.shape[0] == 1:
            return np.sign(ds) * min(abs(ds), ds_max)

        valu, posu = minandmax(state_u)
        ampl_u = np.diff(valu, axis=0)
        vald, posd = minandmax(dirc_u)
        ampl_d = np.diff(vald, axis=0)
        if np.any(np.abs(ampl_d) < 1e-8):
            ds = np.sign(ds) / factor
        else:
            ds = np.sign(ds) * np.min(np.abs(ampl_u / ampl_d)) / factor

        # compute new ds
        while abs(ds) >= ds_max:
            ds /= 2

        return ds.item()

    def orientation(self, ndir, odir):
        # normalize the new direction
        un_dir = ndir.u
        unn = norm(un_dir)
        un_dir /= unn

        # normalize the old direction
        uo_dir = odir.u
        uon = norm(uo_dir)
        uo_dir /= uon

        # also compute the change in da
        da = abs(ndir.a - odir.a) if ndir.n >= 17 else 0.0

        # Can compare to 1 here since both are normalized!
        # If the difference is larger than the normalized norm -> direction flipped!
        # print('norm = ' , norm(uo_dir - un_dir) )
        if norm(uo_dir - un_dir) > 1.0 or da > 1.5 * max(abs(ndir.a), abs(odir.a)):
            return False
        return True

    def solve(self, state, sign=None, direction=None, miter=50, method='nleq',
              preserve_orientation=False, predictor_only=False, *args, **kwargs):
        """ This function implements the core of the basic newton method

        Inputs: state - must be convertible to np.ndarray
                tol   - tolerance of the Newton iteration
                sign  - positive -> parameter must initially increase.
                      - negative -> parameter must initially decrease.

                preserve_orientation - Checks that the input direction and output direction have
                the same orientation. This is to avoid flips which seems to commonly occur when
                dealing with Fredholm operators.

        Notes: 1) Curiously restarting the iteration at a higher discretization size with the
        previous solution leads to linear system convergence failures in roughly 10% of cases.
        """
        success = False
        n = state.n
        m = state.m


        # set shape
        self.shape = (n, m)
        self.function_type = 'trig' if state.u.istrig else 'cheb'

        # Set shape -> important to how to interpret coefficient vectors!
        self.n_eqn = state.shape[1]
        self.cpar = state.cpar
        self.theta = kwargs.get('theta', 0.5)
        self.ntol = kwargs.get('tol', self.ntol)
        ds = kwargs.pop('ds', 0.05)

        # Reset the stored outer_v values
        self.cleanup()

        # make sure theta values match with current!
        state.theta = self.theta

        ###############################
        # Step 0: Compute direction   #
        ###############################
        if direction is None:
            direction = self.solve_direction_svd(state, theta=self.theta, *args, **kwargs)
            # want the direction to be in a certain direction!
            if sign is not None and np.sign(direction.a * ds * np.sign(sign)) < 0:
                # Flip the sign around to match the target!
                # Don't flip s! s must generate a 1-1 map, controlled by the branch itself!
                direction *= -1.0
        else:
            # Since theta may change -> re-normalize the direction
            direction.theta = self.theta
            direction = direction.normalize()

        # store direction -> for callback
        self.direction = direction

        ###############################
        # Step 1: Predictor           #
        ###############################
        un = kwargs.pop('un', None)
        if un is None:
            # Estimate the correct step size!
            # ds = self.estimate_stepsize(state, direction, ds, *args, **kwargs)
            un = self.solve_predictor(state, ds, direction, theta=self.theta, *args, **kwargs)

        # print('Pseudo: a: %.4g -> %.4g; ds = %.4g; θ = %.4g.' % (state.a, un.a, ds, un.theta))
        if self.debug: print('\ta: %.4g -> %.4g; ∫u = %.4g; with ds = %.4g; dir = %s; ' \
                             % (state.a, un.a, np.sum(np.sum(un.u)), ds, direction), un)

        # make sure theta values match!
        un.theta = self.theta

        # If only using Euler steps quit here
        if predictor_only:
            # compute a new direction
            new_dir = self.solve_direction_svd(un, theta=self.theta, *args, **kwargs)
            if preserve_orientation:
                new_dir = self.orientation(new_dir, direction)
            return un, success, new_dir, 1, ds

        ###############################
        # Step 2: Corrector           #
        ###############################
        un, success, iters = self.solve_corrector(un, state, ds, miter=miter,
                                                  direction=direction, theta=self.theta,
                                                  method=method, *args, **kwargs)

        # If linear solver failed we force a direction re-computation
        if self.solve_inner_failure:
            return un, success, None, iters, ds

        ###################################################
        # Step 3: Compute new direction for the next step #
        ###################################################
        # To find a new continuation direction we must resolve LinOp with a new
        # right hand side.
        new_dir, dir_success = self.isolve(np.hstack((np.zeros(np.product(direction.shape)), 1)), x0=np.asarray(direction))
        new_dir.theta = self.theta
        new_dir = new_dir.normalize()

        # If computation of new direction fails return None to indicate that result is not reliable!
        if not dir_success:
            if self.debug: print('Solving for new direction failed!')
            new_dir = self.solve_direction_svd(un, theta=self.theta, *args, **kwargs)

        # Make sure that the direction is not flipped -> if flipped norm should be twice the direction norm
        # print('norm = ', (new_dir - direction).norm(), ' new_dir = ', new_dir.norm())
        # if preserve_orientation and (new_dir - direction).norm() > new_dir.norm():
        # if preserve_orientation:
        #     new_dir = self.orientation(new_dir, direction)

        # Finally return everything
        return un, success, new_dir, iters, ds

    def decel(self):
        if self.contraction == -1.0 and self.norm_dx0 >= 0 and self.angle >= 0:
            f = max(sqrt(self.norm_dx0 / self.nom_distance),
                    self.angle / self.nom_angle)
        elif self.contraction >= 0.0:
            f = max(sqrt(self.contraction / self.nom_curvature),
                    sqrt(self.norm_dx0 / self.nom_distance),
                    self.angle / self.nom_angle)
        else:
            # Didn't see enough iterations -> must be able to increase step size again!
            f = 0.5
        return max(min(f, 2), 0.5)

    @property
    def correct_stepsize(self):
        # Corrector only works when a step-size is known!
        if self.lambda_limitation or self.singular:
            return 0.7
        return sqrt(THETA_BAR / self.thetak)

    @property
    def predict_stepsize(self):
        # If we failed we can't compute anything!
        if not self.newton_success:
            return 0.0

        theta0 = max(0.01, self.thetas[0])
        factor = sqrt(self.norm_dx0 * THETA_BAR / (self.pred_error * theta0 * abs(self.c0)))
        return factor

    def callback(self, itr, dx, normdx, thetak):
        # if itr > 1:
        #     # check whether we can continue!
        #     f = self.decel()
        #     # if f is larger than two then the observed quantities exceed their tolerances
        #     return f < 2

        self.thetas[max(0, itr-1)] = thetak
        self.thetak = thetak

        if itr == 0:
            """ Step size control statistics:

                1) Compute angle between predictor tangent and tangent at first prediction point
                2) Stores the norm of the first correction
                3) Computes the first contraction rate.

            """
            self.norm_dx0 = normdx
            tangent_pred, dir_success = self.isolve(np.hstack((np.zeros(np.product(self.direction.shape)), 1)))
            tangent_pred = tangent_pred.normalize()
            self.c0 = sprod(tangent_pred, self.direction)

            # OLD STUFF
            # if self.dorder == 0:
            #     ip = sprod(self.direction, dx) / normdx
            # elif self.dorder == 1:
            #     normdx = np.sqrt(self.theta * h1norm(dx.u)**2 + (1. - self.theta) * dx.a**2)
            #     normdf = np.sqrt(self.theta * h1norm(self.direction.u)**2 + (1. - self.theta) * self.direction.a**2)
            #     in1 = np.inner(self.direction.u, dx.u)
            #     in2 = np.inner(np.diff(self.direction.u), np.diff(dx.u))
            #     ipu  = np.sum(in1.diagonal()) + np.sum(in2.diagonal())
            #     ip = self.theta * ipu + (1.0 - self.theta) * self.direction.a * dx.a
            #     ip /= (normdx * normdf)
            # self.angle = abs(np.arccos(ip) - 0.5 * np.pi)
        #elif itr == 1:
        #    self.contraction = normdx / self.norm_dx0

        # By default if we don't have enough information continue!
        return True

    def solve2(self, state, sign=None, direction=None, miter=25, method='qnerr',
               max_euler_steps=2, preserve_orientation=False,
               predictor_only=False, flip_direction=False, *args, **kwargs):
        """ This function implements the core of the basic newton method

        Inputs: state - must be convertible to np.ndarray
                tol   - tolerance of the Newton iteration
                sign  - positive -> parameter must initially increase.
                      - negative -> parameter must initially decrease.

                preserve_orientation - Checks that the input direction and output direction have
                the same orientation. This is to avoid flips which seems to commonly occur when
                dealing with Fredholm operators.

        Notes: 1) Curiously restarting the iteration at a higher discretization size with the
        previous solution leads to linear system convergence failures in roughly 10% of cases.
        """
        success = False
        n = state.n
        m = state.m

        # contraction ratio
        self.thetas = np.zeros(miter+2)
        self.thetas[0] = 0.5 * THETA_MAX
        self.rank = 0

        # The predicted solution
        self.pred_pt = None

        # set shape
        self.shape = (n, m)
        self.function_type = 'trig' if state.u.istrig else 'cheb'

        # Set shape -> important to how to interpret coefficient vectors!
        self.n_eqn = state.shape[1]
        self.cpar = state.cpar
        self.theta = kwargs.pop('theta', 0.5)
        self.ntol = kwargs.get('tol', self.ntol)
        self.debug = kwargs.get('debug', self.debug)
        self.verb = kwargs.get('verb', self.verb)
        ds = kwargs.pop('ds', 0.05)

        # Reset the stored outer_v values
        self.cleanup()

        # make sure theta values match with current!
        state.theta = self.theta

        ###############################
        # Step 0: Compute direction   #
        ###############################
        if direction is None:
            direction = self.solve_direction_svd(state, theta=self.theta, *args, **kwargs)
            # want the direction to be in a certain direction!
            if sign is not None and np.sign(direction.a * ds * np.sign(sign)) < 0:
                # Flip the sign around to match the target!
                # Don't flip s! s must generate a 1-1 map, controlled by the branch itself!
                direction *= -1.0
        else:
            # Since theta may change -> re-normalize the direction
            direction.theta = self.theta
            direction = direction.normalize()

        # copy
        s0 = deepcopy(state)

        # since we call main solve several times need to keep track of stats here
        neval = 0
        neitr = 0

        # store direction -> for callback
        self.direction = direction
        direction = direction.prolong(state.n)

        iters = 0
        while True:
            s1 = self.solve_predictor(s0, ds, direction, theta=self.theta, *args, **kwargs)
            self.pred_pt = deepcopy(s1)
            if self.verb: print('\ta: %.4g -> %.4g; ∫u = %.4g; with ds = %.4g; dir = %s; s1 = %s.' \
                                 % (s0.a, s1.a, np.sum(np.sum(s1.u)), ds, direction, s1))

            # make sure theta values match!
            s0.theta = self.theta
            s1.theta = self.theta

            ###############################
            # Step 2: Try Corrector       #
            ###############################
            un, success, its = self.solve_corrector(s1, s0, ds, miter=miter, direction=direction,
                                                    theta=self.theta, method=method, *args, **kwargs)

            # Compute the prediction error
            self.pred_error = (un - self.pred_pt).norm()

            # print('\tdecel-nleq = ', self.decel(), ' pred_error = ', self.pred_error, ' its = ', its, ' success = ', success, ' theta = ', self.theta, ' ds = ', ds)

            # store stats
            neval += self.neval
            neitr += its

            # If it worked we can quit!
            if success and iters > 0:
                print('success after %d euler steps!' % iters)
                s1 = un
                break
            elif success:
                s1 = un
                break
            elif iters >= max_euler_steps:
                # Return old direction?
                self.neval = neval
                return un, False, direction, neitr, ds

            # If things really hit the fan -> some folds try a branch switch
            # ndir = self.swibra(state, direction)

            # The thing we get must have the same orientation!!
            ndir = self.solve_direction_svd(s1, theta=self.theta, *args, **kwargs)

            # flip around if the direction is wrong!
            angle = sprod(ndir, self.direction)
            if angle < 0:
                new_dir = ndir.flip()

            # Check: do I have to do this work?
            angle = sprod(ndir, self.direction)
            if not angle >= 0.3:
                success = False
                return un, False, direction, neitr, ds

            # Relabel -> for next iteration
            direction = ndir
            direction = direction.prolong(state.n)
            s0 = deepcopy(self.pred_pt)
            iters += 1

        # write stats back
        self.neval = neval

        ###################################################
        # Step 3: Compute new direction for the next step #
        ###################################################
        # To find a new continuation direction we must resolve LinOp with a new right hand side.
        #new_dir, dir_success = self.isolve(np.hstack((np.zeros(np.product(direction.shape)), 1)), x0=np.asarray(direction))

        # If computation of new direction fails return None to indicate that result is not reliable!
        dir_success = False
        if not dir_success:
            if self.debug: print('Solving for new direction failed!')
            new_dir = self.solve_direction_svd(s1, theta=self.theta, *args, **kwargs)
        else:
            new_dir.theta = self.theta
            new_dir = new_dir.normalize()

        # compute the determinant
        # self.mdet = self.detFx()

        # flip around if the direction is wrong!
        angle = sprod(new_dir, self.direction)
        if angle < 0:
            #print('Flip Angle not good enough!')
            new_dir = new_dir.flip()

        # Check: do I have to do this work?
        angle = sprod(new_dir, self.direction)
        if not angle >= 0.3:
            #print('Angle not good enough!')
            success = False
            new_dir = None

        # Check that the orientation of the new vector is correct
        # TODO -> add some paranoid check here!!! -> USE LU for null vector -> so that we can save time here!
        # sign, logdet = self.compute_orientation(state=state, direction=new_dir)
        # new_dir.sign = sign

        # Let's be paranoid!
        # if new_dir.sign * direction.sign < 0:
        #     #print('\tFlipping newly computed direction! dir_succss = ', dir_success)
        #     # print('dirc = ', direction)
        #     # print('ndir = ', new_dir)

        # Finally return everything
        return s1, success, new_dir, neitr, ds
