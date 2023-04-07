#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import time
from copy import deepcopy
import numpy as np
import h5py as h5
from fun import Fun, zeros

from colloc.ultraS.ultraS import ultraS
from colloc.trigspec.trigspec import trigspec
from colloc.realDiscretization import realDiscretization
from colloc.source.compiler import Source
from colloc.LinSys import LinSys
from colloc.BiLinSys import BiLinSys
from colloc.function import Function
from colloc.adapt import SolverAdaptive

from newton.newton import Newton
from states.base_state import BaseState
from states.deflation_state import DeflationState


class ChebOp:
    """ A nonlinear operator F : X -> Y where X, Y are Banach spaces.

    Examples:
        1) A linear example where F

                      /  u_xx + u      \
           F[u, v]  = |                | = 0
                      \  v_xx + u + v  /

           subject to

           B[u, v] = (u_x(0), u_x(1), v_x(0), v_x(1)) = 0

        Code example:

           op     = ChebOp(functions=['u', 'v'], domain=[0, 1])
           op.eqn = ['diff(u, x, 2) + u', 'diff(v, x, 2) + u + v']
           op.bcs = [lambda u, v: np.diff(u)(0), lambda u, v: np.diff(u)(1),
                     lambda u, v: np.diff(v)(0), lambda u, v: np.diff(v)(1)]


        2) Nonlinear operators are simply implemented as the linear example above. The nonlinear
           terms are symbolically differentiated using Sympy, and automatically solved using a
           Newton method (NLEQ_ERR + QNERR). If no initial guess is provided the zero function is
           assumed.


        3) Deficient operator problems i.e.\ Fredholm operators of index zero, with non-trivial
           nullspace. These problems require the specification of additional equations which
           remove the operators deficiencies.

        Let F : X -> Y be the Fredholm operator of interest having index zero. We define the
           projection operators

           Q : Y -> Y / R[F]  (i.e. the projection onto the co-kernel of F)
           P : X -> N[F]      (i.e. the projection onto the kernel of F)

        These projections lead to direct sum decompositions of the spaces X and Y i.e.

                X = N[F] + X_2  and   Y = Y / R[F] + R[F]

        Given x ∈ X we decompose

                x = x_1 + x_2  where x_1 ∈ N[F], and x_2 ∈ X_2

        Supposing that we are solving the problem:                F[x] = y
                                                  subject to:     M[x] = z

            where M : X -> Y is such that it removes the deficiencies of F. That is we require that
            R[M] = Y / R[F] and N[M] = N[P]

        We solve the resulting problem in two steps:

            1) F[x_2] = (I - Q)[y]      # An operator equation between X_2 -> R[F]
            2) M[x_1] = Q[z]            # An operator equation between N[F] -> Y / R[F]

        TODO: There are projections missing in the last equation!

        Since the above two equations are defined on the two direct summands of X and Y we can add
        the two equations together yielding our final linear algebra problem:

            M[x_1] + F[x_2] = (I - Q)[y] + Q[z]

        Note that this operator is no longer deficient!
    """
    def __init__(self, colloc='ultraS', *args, **kwargs):
        # local namespace
        self.debug = kwargs.pop('debug', False)
        self.run_debug = kwargs.pop('run_debug', False)
        self.solver_debug = kwargs.pop('solver_debug', False)

        self.function_names = kwargs.pop('functions', [])
        self.constant_function_names = kwargs.pop('constant_functions', {})
        self.operator_names = kwargs.pop('operators', [])
        self.kernels = kwargs.pop('kernels', [])
        self.parameter_names = kwargs.pop('parameters', {})
        self.colloc = colloc
        self.colloc_type = dict(ultraS=ultraS,
                                trigspec=trigspec).get(colloc, colloc)
        self.domain = kwargs.pop('domain', [-1, 1])

        # function type
        self.ftype = kwargs.pop('ftype', 'cheb' if self.colloc == 'ultraS' else 'trig')

        # self.functions = [u, v]
        # The default discretization size!
        self.n_disc = kwargs.pop('n', 15)
        self.n_min = kwargs.pop('n_min', 1 if self.ftype == 'cheb' else 2**3+1)

        # The expression for the equation.
        self.eqn = kwargs.pop('eqn', [])

        # Equation constraints
        self.cts = kwargs.pop('constraints', [])

        # The equations boundary conditions.
        self.bcs = kwargs.pop('bc', [])

        # store cpar
        self.cpar = kwargs.pop('cpar', [])

        # Projections
        self.proj = []

        # operator eps -> TODO make this settable
        self.eps = kwargs.pop('eps', 2e3 * np.finfo(float).eps)

        # tolerance for the linear solver
        self.lin_tol = kwargs.pop('tol', 1e-2 * self.eps)

        # get output space
        self.diffOrder = kwargs.pop('diff_order', 2)

        # is the operator linear?
        self.linear = kwargs.pop('linear', False)

        # linear system
        self.linSys   = None
        self.__biLinSys = None
        self.__mooreSys = None
        self.__biLinSysAction = None

        # operator source
        self.source = None

        # Store the operator's newton method
        self.newton = kwargs.pop('newton', None)

        # prepared
        self.prepared = False
        self.par_set = False

        # is linear
        self.isLinear = False

        # non-local
        self.isNonLocal = True if len(self.operator_names) > 0 else False

    @property
    def islinear(self):
        return self.linear or np.all(self.isLinear)

    @property
    def shape(self):
        return (len(self.eqn), len(self.eqn))

    @property
    def neqn(self):
        return len(self.eqn)

    @property
    def pars(self):
        if self.linSys is not None: return self.linSys.ns['print_pars']()
        else: return self.parameter_names

    @property
    def biLinSys(self):
        # TODO: should be able to do this more elegantly!
        # If it's created yet generated it!
        if self.source is None: return self.__biLinSys
        if self.__biLinSys is None: self.compile_bilin(par=True, debug=self.debug)
        return self.__biLinSys

    @property
    def biLinSys_action(self):
        if self.source is None: return self.__biLinSysAction
        #if self.__biLinSysAction is None:
        self.compile_action(par=True, debug=self.debug)
        return self.__biLinSysAction

    @property
    def mooreSys(self):
        # TODO: should be able to do this more elegantly!
        # If it's created yet generated it!
        if self.source is None: return self.__mooreSys
        if self.__mooreSys is None: self.compile_moore(par=True, debug=self.debug)
        return self.__mooreSys

    def to_state(self, u, cpar, theta=0.5):
        return DeflationState(a=self.parameter_names[cpar], u=u, theta=theta,
                              n=self.n_disc, cpar=cpar, **self.parameter_names)

    def setDisc(self, n):
        n = int(n)
        self.n_disc = max(n, self.n_min)
        if self.linSys is not None: self.linSys.setDisc(n)
        if self.__biLinSys is not None: self.biLinSys.setDisc(n)
        if self.__mooreSys is not None: self.mooreSys.setDisc(n)

    def __str__(self):
        rstr = 'ChebOp['
        first = True
        for p_name, p_value in self.parameter_names.items():
            if not first: rstr += '; '
            rstr += '%s = %.4g' % (p_name, p_value)
            first = False
        rstr += ']'
        return rstr

    def __deepcopy__(self, memo):
        id_self = id(self)
        _copy = memo.get(id_self)
        if _copy is None:
            _copy = type(self)(
                debug=self.debug,
                functions=deepcopy(self.function_names),
                constant_functions=deepcopy(self.constant_function_names),
                operator=deepcopy(self.operator_names),
                kernels=deepcopy(self.kernels),
                parameters=deepcopy(self.parameter_names),
                colloc=self.colloc,
                domain=self.domain,
                n_disc=self.n_disc,
                eqn=deepcopy(self.eqn),
                constraints=deepcopy(self.cts),
                bc=deepcopy(self.bcs),
                ftype=self.ftype,
                cpar=self.cpar,
                tol=self.lin_tol,
                diff_order=self.diffOrder,
                linear=self.linear
            )
            memo[id_self] = _copy
        return _copy

    def compile(self, par=False, fold=False, bif=False, *args, **kwargs):
        # If we have the base setup let's not call this again!
        if self.prepared and (self.par_set or par == self.par_set):
            return

        # Compile the source!
        debug = kwargs.get('debug', self.debug)
        constant_functions = kwargs.pop('constant_functions', False)
        self.par_set = par
        # update this!
        self.isNonLocal = True if len(self.operator_names) > 0 else False

        # Create source object and generate pycode
        self.source = Source(self.eqn, self.cts, self.bcs,
                             self.function_names,
                             self.constant_function_names,
                             self.operator_names, self.parameter_names,
                             self.kernels, nonLocal=self.isNonLocal,
                             proj=self.proj, ftype=self.ftype,
                             constant_functions=constant_functions,
                             domain=self.domain)

        # Do the sympy compile and generate the required operator python code!
        self.source.compile(debug=self.debug, par=par, fold=par,
                            bif=bif, cpars=self.cpar)

        # Set diffOrder
        self.diffOrder = self.source.diffOrder

        # Create Linear System object now
        self.linSys = LinSys(self.source, self.diffOrder, n_disc=self.n_disc,
                             par=par, debug=debug)

        # Set parameters
        self.linSys.setParameters(self.parameter_names)

        # set prepared
        self.prepared = True

    def compile_action(self, debug=False, par=False, *args, **kwargs):
        # Create Linear System object now
        self.__biLinSysAction = Function(self.source, n_disc=self.n_disc, debug=debug)

        # Set parameters
        self.__biLinSysAction.setParameters(self.parameter_names)

    def compile_bilin(self, debug=False, par=False, *args, **kwargs):
        # Create Linear System object now
        # TODO: auto set diffOrder
        self.__biLinSys = BiLinSys(self.source, 0, n_disc=self.n_disc, projOrder=2,
                                   par=par, debug=debug, matrix_name='fold')

        # Construct the functional constraints
        #self.__biLinSys.constraints = [ChebOpConstraint(op=bc, domain=self.domain) for bc in self.bcs]

        # Set parameters
        self.__biLinSys.setParameters(self.parameter_names)

    def compile_moore(self, debug=False, par=False, *args, **kwargs):
        # Create Linear System object now
        # TODO: auto set diffOrder
        self.__mooreSys = BiLinSys(self.source, 0, n_disc=self.n_disc, projOrder=2,
                                   par=par, debug=debug, matrix_name='bif', dp_name='dxdp_adj')

        # Construct the functional constraints
        #self.__mooreSys.constraints = [ChebOpConstraint(op=bc, domain=self.domain) for bc in self.bcs]

        # Set parameters
        self.__mooreSys.setParameters(self.parameter_names)

    def solve(self, f=None, p=2, verbose=True, state=False,
              adaptive=True, *args, **kwargs):
        """
        This function solves Op[u] = f using Newton's method.

        Arguments:
            state - boolean "If true we return an object of type State "
        """
        # Use a Newton's method to solve the nonlinear operator
        debug = self.solver_debug or kwargs.pop('debug', False)
        eps = kwargs.pop('eps', self.eps)
        newton = Newton(self, tol=eps, *args, **kwargs)
        newton.shape = (self.n_disc, len(self.eqn))

        # Create initial state
        if f is None: f = zeros(len(self.eqn), domain=self.domain)
        i_state = f if isinstance(f, BaseState) else DeflationState(u=f, **self.parameter_names)
        if self.linSys is not None: self.linSys.setParameters(i_state)

        # Call the solver -> start the timer
        solve_st = time.time()

        if adaptive:
            anw = SolverAdaptive(newton, n_min=self.n_min, ntol=eps)
            s_state, success, iterations = anw.solve(
                i_state, eps=1e2*eps, debug=debug,
                verbose=verbose, *args, **kwargs)
        else:
            precond = False if self.ftype == 'trig' else True
            s_state, success, iterations = newton.solve(
                i_state, precond=precond, debug=debug, *args, **kwargs)

        # The core solver is done -> stop the timer
        solve_ed = time.time()

        # TODO: Can we do this better?
        try:
            soln = s_state.u
        except AttributeError:
            soln = s_state

        # If solver failed return None
        if soln is None:
            return None, False, np.inf

        # Compute the residual
        res = self.residual(soln)

        # simplify solution
        happy, _ = soln.happy()
        print('happy = ', happy, ' shape = ', soln.shape)
        soln = soln.simplify(eps=np.finfo(float).eps)
        print('soln = ', soln.shape)

        # Report whether the solve was successful!
        message = {True: 'successful', False: 'failed'}
        nonlin  = {False: 'Non-linear', True: 'Linear'}

        # Some consoles don't have unicode support!
        try:
            print('{0:s} operator solved {1:s}. |res|{2:d} = {3:.6e}; ∫u = {4:.6e}.'.format(
                nonlin[self.islinear], message[success], p, res, np.sum(np.sum(soln))))
        except:
            print('{0:s} operator solved {1:s}. |res|{2:d} = {3:.6e}; Iu = {4:.6e}.'.format(
                nonlin[self.islinear], message[success], p, res, np.sum(np.sum(soln))))

        print('\tSolver time:', solve_ed - solve_st)

        if state:
            return s_state, success, res
        else:
            return soln, success, res

    def __call__(self, u):
        self.linSys.setDisc(u.shape[0])
        self.linSys.update_partial(u)
        return Fun(coeffs=self.linSys.rhs.values, domain=u.domain,
                   simplify=False, type=self.ftype)

    def residual(self, u, p=2):
        self.linSys.update_partial(u)
        res = Fun(coeffs=self.linSys.rhs.values, domain=u.domain, type=self.ftype)
        residual = res.norm(p=p)

        if self.linSys.numConstraints > 0:
            residual += np.linalg.norm(self.linSys.cts_res, ord=p)

        return residual

    def matrix(self, *args, **kwargs):
        colloc = self.discretize(*args, **kwargs)
        return colloc.matrix()

    def matrix_nonproj(self, *args, **kwargs):
        colloc = self.discretize(*args, **kwargs)
        return colloc.matrix_nonproj()

    def discretize_system(self, operator, u0, par=False, *args, **kwargs):
        # Generate the appropriate quasi-matrices!
        operator.setDisc(self.n_disc)

        # Generate the quasimatrix of the linearised operator at u0
        operator.quasi(u0, eps=u0.eps)
        if self.run_debug: print(operator.pars)

        if self.n_disc == 1:
            if self.run_debug: print('Real discretization!')
            return realDiscretization(operator, domain=self.domain, par=par)
        else:
            if self.run_debug: print('Collocation discretization!')
            return self.colloc_type(operator, domain=self.domain, par=par, dimension=self.n_disc)

    def discretize(self, u0, sys_name='linear', *args, **kwargs):
        """ Returns the discretized version of the linear version of the operator """
        # make sure domains match
        assert np.all(self.domain == u0.domain), 'Domain mismatch %s != %s!' % (self.domain, u0.domain)

        # system options
        system_choices = {'linear': 'linSys',
                          'fold': 'biLinSys',
                          'bif': 'mooreSys'}

        assert sys_name in system_choices.keys(), '{} is an unknown system!'.format(sys_name)

        # make sure that the operator is prepared!
        self.ftype = u0.type
        par = kwargs.pop('par', False)
        par = False if not self.cpar else True
        self.compile(ftype=u0.type, par=par, *args, **kwargs)

        return self.discretize_system(getattr(self, system_choices[sys_name]), u0, *args, **kwargs)

    """ I/O support """
    def writeHDF5(self, fh):
        dgrp = fh.create_group(type(self).__name__)

        # Write function names
        utf8_type = h5.string_dtype('utf-8', 30)
        dgrp['function_names'] = np.asarray([s.encode('utf-8') for s in self.function_names], dtype=utf8_type)
        dgrp['constant_function_names'] = np.asarray([s.encode('utf-8') for s in self.constant_function_names], dtype=utf8_type)
        dgrp['operator_names'] = np.asarray([s.encode('utf-8') for s in self.operator_names], dtype=utf8_type)
        dgrp['kernels'] = np.asarray([s.encode('utf-8') for s in self.kernels], dtype=utf8_type)
        par_grp = dgrp.create_group('parameters')
        for pname, pvalue in self.parameter_names.items():
            par_grp.attrs[pname.encode('utf-8')] = pvalue

        dgrp['colloc'] = np.asarray(self.colloc.encode('utf-8'), dtype=utf8_type)

        utf8_type = h5.string_dtype('utf-8', 250)
        dgrp['eqn'] = np.asarray([s.encode('utf-8') for s in self.eqn], dtype=utf8_type)
        dgrp['cts'] = np.asarray([s.encode('utf-8') for s in self.cts], dtype=utf8_type)
        dgrp['bcs'] = np.asarray([s.encode('utf-8') for s in self.bcs], dtype=utf8_type)

        # parameters
        dgrp['debug'] = self.debug
        dgrp['solver_debug'] = self.solver_debug
        dgrp['n_disc'] = self.n_disc
        dgrp['n_min'] = self.n_min
        dgrp['ftype'] = self.ftype
        dgrp['cpar'] = self.cpar[0]
        dgrp['domain_beg'] = self.domain[0]
        dgrp['domain_end'] = self.domain[1]
        dgrp['eps'] = self.eps
        dgrp['lin_tol'] = self.lin_tol
        dgrp['diffOrder'] = self.diffOrder
        dgrp['linear'] = self.linear

    def readHDF5(self, fh):
        dgrp = fh

        for fname in fh['function_names']:
            self.function_names.append(fname.decode('utf-8'))

        for oname in fh['operator_names']:
            self.operator_names.append(fname.decode('utf-8'))

        for kernel in fh['kernels']:
            self.kernels.append(kernel.decode('utf-8'))

        for pname, pvalue in fh['parameters'].attrs.items():
            self.parameter_names[pname] = pvalue

        self.colloc = fh['colloc'][()]

        for eq in fh['eqn']:
            self.eqn.append(eq.decode('utf-8'))

        for ct in fh['cts']:
            self.cts.append(ct.decode('utf-8'))

        try:
            for bc in fh['bcs']:
                self.bcs.append(bc.decode('utf-8'))
        except:
            pass

        # parameters
        self.debug = dgrp['debug'][()]
        self.solver_debug = dgrp['solver_debug'][()]
        self.n_disc = dgrp['n_disc'][()]
        self.n_min = dgrp['n_min'][()]
        self.eps = dgrp['eps'][()]
        self.lin_tol = dgrp['lin_tol'][()]
        self.diffOrder = dgrp['diffOrder'][()]
        self.linear = dgrp['linear'][()]
        try:
            self.cpar = [dgrp['cpar'][()]]
            self.colloc = [dgrp['colloc'][()]]
            self.ftype = dgrp['ftype'][()]
        except:
            pass

        try:
            self.domain[0] = dgrp['domain_beg'][()]
            self.domain[1] = dgrp['domain_end'][()]
        except:
            pass
