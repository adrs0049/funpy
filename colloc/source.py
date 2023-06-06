#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
import sympy as syp
import datetime

from sympy import Function
from sympy.matrices import zeros, ones
# from sympy import sympify, latex, pprint
# from IPython.display import display

from ac.support import Namespace
from ac.gen import CodeGeneratorBackend

from cheb.cbcode import cpcode
# from cheb.odecode import odecode
from .source.support import sympy_base_program, convolve
from .tools import execute_pycode


class Source:
    """ Source for the discretization of an operator on a Hilbert space.

        The required Frechet derivatives are computed symbolically using SymPy.

        TODO: Sometime in the future improve the quality and modularization of
        the code below. Right now this is completely horrific.
    """
    def __init__(self, eqn_src, cts_src, function_names, operator_names,
                 parameter_names, integration_kernels, *args, **kwargs):
        self.ns = Namespace()

        self.diffOrder = kwargs.pop('diffOrder', 2)

        self.eqn_src = eqn_src
        self.cts_src = cts_src
        self.ftype = kwargs.pop('ftype', 'cheb')

        self.function_names = function_names
        self.operator_names = operator_names
        self.integration_kernels = integration_kernels
        self.parameter_names = parameter_names
        self.constant_functions = kwargs.pop('constant_functions', False)
        self.domain = kwargs.pop('domain', [-1, 1])
        self.proj_src = kwargs.pop('proj', [])

        # Keep track of the compiled parameters
        self.compiled_pars = []

        self.syp_src = {}
        self.symbol_names = {'eqn': 'f', 'cts': 'c', 'proj': 'proj', 'adh': 'g',
                             'bif': 'moore', 'fold': 'fold'}

        # non-local operators have additional terms showing up in the linearization
        self.isNonLocal = kwargs.pop('nonLocal', False)
        if self.isNonLocal:
            assert len(self.operator_names) > 0, 'Non-local source requires non-local operator definition!'

        self.functions = None   # Sympy function objects
        self.operators = None   # Sympy operator objects
        self.kernels = None     # Sympy integration kernel objects
        self.common = None      # Contains common python code
        self.eqn  = None        # Compiled functions for the residual
        self.cts  = None        # Compiled functions for the constraints
        self.dp   = ''          # Compiled functions for the dp
        self.dxdp = ''          # Compiled functions for the dxdp
        self.dxdp_adj = ''      # Compiled functions for the adjoint of dxdp
        self.dpdp_adj = ''      # Compiled functions for the adjoint of dxdp

        # Compiled code
        self.d_spcode = {}
        self.dmdp = {}
        self.dmdxdp = {}
        self.dmdxdp_adj = {}
        self.dmdpdp_adj = {}
        self.par_pert = {}

        # Required temporary information
        self.pfuns = []
        self.pfuns2 = []
        self.ppfunc_names = []
        self.pfunc_names = []

    @property
    def lin(self):
        return self.syp_src['deqn']

    @property
    def bif(self):
        return self.syp_src['bif']

    @property
    def fold(self):
        return self.syp_src['fold']

    @property
    def lin_cts(self):
        if not self.cts_src: return None
        return self.syp_src['dcts']

    @property
    def rhs(self):
        return self.eqn

    def compile(self, cpar=[], debug=False, par=False):
        self.init(debug=debug)
        self.functions = [self.ns[fname] for fname in self.function_names]
        self.operators = [self.ns[oname] for oname in self.operator_names]
        self.kernels   = [self.ns[kname] for kname in self.integration_kernels]

        # store the parameter names for which compiled stuff
        if not isinstance(cpar, list):
            cpar = [cpar]
        self.compiled_pars += cpar

        # Create common code
        pycode = self.__emit_common()
        if debug: print(pycode)
        self.common = pycode

        # compile right hand side
        for type in ['eqn', 'cts']:
            src = getattr(self, type + '_src')
            if src is None: continue
            name = self.symbol_names[type]
            spcode = self.sympify(src, name, debug=debug)
            self.syp_src[type] = spcode

            # Generate python code
            pycode = self.__emit_function(spcode, name=name, debug=debug)
            if debug: print(pycode)
            setattr(self, type, pycode)

            # Generate linearization
            d_spcode = self.linearize(spcode, debug=debug)
            self.d_spcode[type] = d_spcode
            pycode = self.compute_coefficients(d_spcode, self.diffOrder, name=name, debug=debug)
            dname = 'd' + type
            self.syp_src[dname] = pycode

        if par:  # Compile bifurcation diagram support!
            # Compile equation derivatives w.r.t. parameters
            self.compile_pars(cpar, debug=debug)

            # Generate the linearization of F'(y)^T z; where z in R[A]
            self.dd_spcode = self.second_order_derivative(self.d_spcode['eqn'], debug=debug)

            # Generate python code for the second order derivative
            pycode = self.compute_coefficients2(self.dd_spcode, 0,
                                                name=self.symbol_names['fold'], debug=debug)
            self.syp_src['fold'] = pycode

            # Compute a bunch of coefficients
            self.compute_dxdp(self.dM, cpar, debug=debug)

            # Generate python code for the second order derivative
            self.dd_spcode_adjoint = self.second_order_derivative_adjoint(self.d_spcode['eqn'], debug=debug)
            pycode = self.compute_coefficients2(self.dd_spcode_adjoint, 0,
                                                name=self.symbol_names['bif'], debug=debug)
            self.syp_src['bif'] = pycode

            # Compute a bunch of coefficients
            self.compute_dxdp_adjoint(self.dMT, cpar, debug=debug)
            self.compute_dxdp_adjoint_par(self.dmdp, cpar, debug=debug)

    def compile_pars(self, cpars, debug=False):
        cg = CodeGeneratorBackend()
        cg.begin(tab=4*" ")
        cg.write(50 * '#')
        cg.write('# Derivative of F(y) w.r.t. parameters block.')
        cg.write(50 * '#')
        self.dp += cg.end()

        for cpar in cpars:
            self.dp += self.eqn_parameter_derivatives(self.syp_src['eqn'], cpar, debug=debug)

        # print code if required!
        if debug: print(self.dp)

    def compute_dxdp(self, matrix, cpars, debug=False):
        """ Computes the derivative of F'(y) [phi] with respect to the equation's parameters. """
        # Write the header!
        cg = CodeGeneratorBackend()
        cg.begin(tab=4*" ")
        cg.write(50 * '#')
        cg.write('# Derivative of F\'(y)[φ] w.r.t. parameters block.')
        cg.write('# y = (u, v); φ = (u_p, v_p).')
        cg.write(50 * '#')
        self.dxdp += cg.end()

        for cpar in cpars:
            self.dmdxdp[cpar] = self.dx_parameter_derivatives(matrix, cpar, debug=debug)
            self.dxdp += self.__emit_dxdp(self.dmdxdp[cpar], cpar)

        # print code if required!
        if debug: print(self.dxdp)

    def compute_dxdp_adjoint(self, matrix, cpars, debug=False):
        """ Computes the derivative of F'(y) [phi] with respect to the equation's parameters. """
        # Write the header!
        cg = CodeGeneratorBackend()
        cg.begin(tab=4*" ")
        cg.write(50 * '#')
        cg.write('# Derivative of F\'(y)^T [φ] w.r.t. parameters block.')
        cg.write('# y = (u, v); φ = (u_p, v_p).')
        cg.write(50 * '#')
        self.dxdp_adj += cg.end()

        for cpar in cpars:
            self.dmdxdp_adj[cpar] = self.dx_parameter_derivatives(matrix, cpar, debug=debug)
            self.dxdp_adj += self.__emit_dxdp(self.dmdxdp_adj[cpar], cpar, func_name='dxdp_adj')

        # print code if required!
        if debug: print(self.dxdp_adj)

    def compute_dxdp_adjoint_par(self, matrix, cpars, debug=False):
        """ Computes the derivative of F'(y) [phi] with respect to the equation's parameters. """
        # Write the header!
        cg = CodeGeneratorBackend()
        cg.begin(tab=4*" ")
        cg.write(50 * '#')
        cg.write('# Derivative of F\'(λ)^T [φ] w.r.t. all.')
        cg.write('# y = (u, v); φ = (u_p, v_p).')
        cg.write(50 * '#')
        self.dpdp_adj += cg.end()

        for cpar in cpars:
            self.dmdpdp_adj[cpar] = self.dp_derivatives(matrix[cpar], cpar, debug=debug)
            self.dpdp_adj += self.__emit_dpdp(self.dmdpdp_adj[cpar], cpar, func_name='dpdp_adj')

        # print code if required!
        if debug: print(self.dpdp_adj)

    def dp_derivatives(self, src, p_name, debug=False):
        # Assemble the output of F'(y) [z]
        assert p_name in self.parameter_names.keys(), '%s not a known parameter!' % p_name
        p = self.ns[p_name]
        n_src = src.shape[0]

        # Compute the action of F'(y) [z]
        self.dP = zeros(1, 1)
        for i in range(n_src):
            self.dP[0, 0] += src[i, 0] * self.pfuns[i]

        # perturbation size
        e  = syp.symbols('eps_pp', real=True)

        # Need to group values
        DM = zeros(n_src + 1, 1)
        for i in range(n_src):
            exp = self.dP[0, 0].replace(self.functions[i], self.functions[i] + e * self.pfuns2[i])

            # Compute the partial Frechet derivative!
            # Make sure to call expand here -> otherwise subsequent coeff might fail!
            exp = syp.expand(syp.collect(syp.diff(exp, e).subs({e: 0}), self.pfuns2[i]))

            # Do the coefficient extraction here already!
            DM[i, 0] = syp.simplify(exp.coeff(self.pfuns2[i]))

        # Define the perturbation parameter name
        p_pert_name = str(p)+'_pp'
        pp = syp.symbols(p_pert_name, real=True)

        # compute parameter perturbation
        exp = self.dP[0, 0].replace(p, p + e * pp)

        # Compute the partial Frechet derivative!
        # Make sure to call expand here -> otherwise subsequent coeff might fail!
        DM[n_src, 0] = syp.expand(syp.collect(syp.diff(exp, e).subs({e: 0}), pp))
        return DM

    def dx_parameter_derivatives(self, src, p_name, debug=False):
        # Assemble the output of F'(y) [z]
        assert p_name in self.parameter_names.keys(), '%s not a known parameter!' % p_name
        p = self.ns[p_name]
        n_src = src.shape[0]

        # Define the perturbation parameter name
        p_pert_name = str(p)+'_pp'

        # The increment used to compute the Frechet derivative
        e  = syp.symbols('eps_pp', real=True)
        pp = syp.symbols(p_pert_name, real=True)
        self.par_pert[p_name] = pp

        # Need to group values
        DM = zeros(n_src, 1)
        for i in range(n_src):
            exp = src[i, 0].replace(p, p + e * pp)

            # Compute the partial Frechet derivative!
            # Make sure to call expand here -> otherwise subsequent coeff might fail!
            DM[i, 0] = syp.expand(syp.collect(syp.diff(exp, e).subs({e: 0}), pp))

        return DM

    def second_order_derivative(self, matrix, debug=False):
        """ Computes the second derivatives of the function F(y) where y = [function_names].

        In all applications we don't require the full tensor; but a linear operator

            D^2 F[phi][w]

        so we prepare for this particular case.
        """
        # Assemble the output of F'(y) [z]
        n_src = matrix.shape[0]
        m_src = matrix.shape[1]

        # We want to perturb once more! -> use new function names.
        x = self.ns['x']
        self.ppfunc_names = [str(fun) + '_pp' for fun in self.function_names]
        self.pfuns2 = [Function(fname)(x) for fname in self.ppfunc_names]

        # Symbolically compute the action of F'(y)[z]
        self.dM = zeros(n_src, 1)
        temp_vector = ones(rows=1, cols=m_src)
        for i in range(n_src):
            self.dM[i] = matrix.row(i).dot(temp_vector)

        # The increment used to compute the Frechet derivative
        e = syp.symbols('eps_pp', real=True)

        # Need to group values
        DM = zeros(n_src, m_src)
        for i, j in np.ndindex(DM.shape):
            fun, pfun = self.functions[j], self.pfuns2[j]
            exp = self.dM[i, 0].replace(fun, fun + e * pfun)

            # Compute the partial Frechet derivative!
            # Make sure to call expand here -> otherwise subsequent coeff might fail!
            DM[i, j] = syp.simplify(syp.collect(syp.diff(exp, e).subs({e: 0}), pfun))

        return DM

    def second_order_derivative_adjoint(self, matrix, debug=False):
        """ Computes the second derivatives of the function F(y) where y = [function_names].

        In all applications we don't require the full tensor; but a linear operator

            D( D F^T [phi] )[w]

        so we prepare for this particular case.
        """
        # Assemble the output of F'(y) [z]
        n_src = matrix.shape[0]
        m_src = matrix.shape[1]

        # We want to perturb once more! -> use new function names.
        x = self.ns['x']
        self.ppfunc_names = [str(fun) + '_pp' for fun in self.function_names]
        self.pfuns2 = [Function(fname)(x) for fname in self.ppfunc_names]


        # Take transpose -> need to replace some of the function names
        matrix_transpose = zeros(m_src, n_src)
        for i, j in np.ndindex(matrix.shape):
            if i == j: matrix_transpose[i, j] = matrix[i, j]
            else: matrix_transpose[i, j] = matrix[j, i].replace(self.pfuns[i], self.pfuns[j])

        # Symbolically compute the action of F'(y)^T [z]
        self.dMT = zeros(n_src, 1)
        temp_vector = ones(rows=1, cols=m_src)
        for i in range(n_src):
            self.dMT[i] = matrix_transpose.row(i).dot(temp_vector)

        # The increment used to compute the Frechet derivative
        e = syp.symbols('eps_pp', real=True)

        # Need to group values
        DMT = zeros(m_src, n_src)
        for i, j in np.ndindex(DMT.shape):
            fun, pfun = self.functions[j], self.pfuns2[j]
            exp = self.dMT[i, 0].replace(fun, fun + e * pfun)

            # Compute the partial Frechet derivative!
            # Make sure to call expand here -> otherwise subsequent coeff might fail!
            DMT[i, j] = syp.simplify(syp.collect(syp.diff(exp, e).subs({e: 0}), pfun))

        return DMT

    def linearize(self, matrix, debug=False):
        """ Computes the linearization of the operator stored in matrix """
        # replace each of the functions with the perturbation
        x = self.ns['x']
        self.pfunc_names = [str(fun) + '_p' for fun in self.function_names]
        self.pfuns = [Function(fname)(x) for fname in self.pfunc_names]
        pfuns = self.pfuns

        # The increment used to compute the Frechet derivative
        e = syp.symbols('eps_p', real=True)

        # Need to group values
        n = matrix.shape[0]
        m = len(self.functions)
        DM = zeros(n, m)

        # look up the sympy objects
        funs = self.functions
        ops  = self.operators
        assert len(ops) <= 1, 'Can only handle at most one operator!'

        if self.isNonLocal:
            kOp = ops[0]   # Assuming only one operator for the moment
            op_name = self.operator_names[0]
            kOp_p = Function(str(op_name)+'_p')(x)
        # else -> these symbols should not be accessed!

        for i, j in np.ndindex(DM.shape):
            fun, pfun = funs[j], pfuns[j]
            # If non-local -> also need to replace non-local operator!
            if self.isNonLocal:
                # For the moment only deal with a single operator
                exp = matrix[i, 0].replace(fun, fun + e * pfun).replace(kOp, kOp + e * kOp_p)
            else:
                exp = matrix[i, 0].replace(fun, fun + e * pfun)

            # Compute the partial Frechet derivative!
            # Make sure to call expand here -> otherwise subsequent coeff might fail!
            DM[i, j] = syp.expand(syp.collect(syp.diff(exp, e).subs({e: 0}), pfun))

            # If non-local operator we must now substitute in the correct definitions!
            if self.isNonLocal:
                # assuming we have only one integration kernel
                kernel = self.kernels[0]
                # Create the expressions for the adhesion terms
                exprK = convolve(fun,  kernel, x, lower_limit=-1, upper_limit=1)
                exprKp = convolve(pfun, kernel, x, lower_limit=-1, upper_limit=1)
                DM[i, j] = DM[i, j].replace(kOp, exprK).replace(kOp_p, exprKp)

        return DM

    def compute_coefficients2(self, dF, diffOrder, space='x', name='f', debug=False):
        # TODO: This function has become way too complicated -> Simplify me!
        # The matrix shape must also include constraints that may be defined in addition!
        matrix = np.empty(dF.shape, dtype=object)
        x = self.ns['x']

        # x, u, v -> to lambda x: -> for Fun computation
        function_names = []
        function_names += self.function_names
        function_names += self.pfunc_names

        # Generate code to compute the coefficients of the linear operator
        for i, j in np.ndindex(dF.shape):
            pfun = self.pfuns2[j]
            exp = dF[i, j]

            # code gen
            cg = CodeGeneratorBackend()
            cg.begin(tab=4*" ")
            cg.write(45 * '#')
            cg.write('# Derivative block {0:s}[{1:d}, {2:d}] out of [{3:d}, {4:d}].'.\
                     format(name, i, j, dF.shape[0], dF.shape[1]))
            cg.write(45 * '#')

            # store positivity information for later lookup
            positivity_information = np.ones(diffOrder + 2).astype(bool)

            order = 0
            while order <= diffOrder:
                # Only call simplify when we have no non-local problems ->
                #   messes up integral derivative ordering.
                ex = syp.simplify(exp.coeff(syp.Derivative(pfun, (x, order))))

                # these pieces of code need to be callable
                # x, u, v -> to lambda x: -> for Fun computation
                ccode = cpcode(ex, function_names=function_names,
                               allow_unknown_functions=True, no_evaluation=True)

                # Write to the code gen
                cg.write('def {0:s}{1:d}{2:d}{3:d}({4}):'\
                         .format(name, i, j, order, ', '.join(map(str, function_names))))
                cg.indent()

                # Depending on the type of expression we need to write the numpy code differently!
                real = ex.is_real if ex.is_real is not None else False
                if ex.is_zero:
                    cg.write('return zeros(1, domain=[{0:.16f}, {1:.16f}], type=\'{2:s}\')'\
                             .format(self.domain[0], self.domain[1], self.ftype))
                    positivity_information[order] = False
                elif ex.is_constant() or real:
                    cg.write('return {0} * ones(1, domain=[{1:.16f}, {2:.16f}], type=\'{3:s}\')'\
                             .format(ccode, self.domain[0], self.domain[1], self.ftype))
                else:
                    cg.write('return {0}'.format(ccode))

                cg.write('')
                cg.dedent()
                order += 1

            # Check whether we have any integral terms!
            ex = simplify(exp.coeff(syp.Integral(pfun)))
            cg.write('def {0:s}{1:d}{2:d}{3:s}({4}):'.format(name, i, j, 'i',
                                                             ', '.join(map(str, function_names))))
            cg.indent()

            if not ex.is_zero:
                real = ex.is_real if ex.is_real is not None else False
                # Compile the code into something useful
                ccode = cpcode(ex, function_names=function_names,
                               allow_unknown_functions=True, no_evaluation=True)
                if ex.is_constant() or real:
                    cg.write('return {0} * ones(1, domain=[{1:.16f}, {2:.16f}], type=\'{3:s}\')'.format(ccode, self.domain[0], self.domain[1], self.ftype))
                else:
                    cg.write('return {0}'.format(ccode))
            else:  # nothing -> write zero
                positivity_information[-1] = False
                cg.write('return zeros(1, domain=[{0:.16f}, {1:.16f}], type=\'{2:s}\')'.format(self.domain[0], self.domain[1], self.ftype))
            cg.dedent()

            # create code for positivity information
            cg.write('')
            cg.write('# Return sorted by derivative term order, followed by the integral terms.')
            cg.write('def {0:s}{1:s}{2:d}{3:d}():'.format(name, 'info', i, j))
            cg.indent()
            cg.write('return [{0}]'.format(', '.join(map(str, positivity_information))))
            cg.dedent()

            # get the code block
            matrix[i, j] = cg.end()
            if debug: print(matrix[i, j])

        return matrix

    def compute_coefficients(self, dF, diffOrder, space='x', name='f', nonlocal_name='g', debug=False):
        # TODO: This function has become way too complicated -> Simplify me!
        # The matrix shape must also include constraints that may be defined in addition!
        matrix = np.empty(dF.shape, dtype=object)
        x = self.ns['x']

        # Generate code to compute the coefficients of the linear operator
        for i, j in np.ndindex(dF.shape):
            pfun = self.pfuns[j]
            exp = dF[i, j]

            # code gen
            cg = CodeGeneratorBackend()
            cg.begin(tab=4*" ")
            cg.write(45 * '#')
            cg.write('# Derivative block {0:s}[{1:d}, {2:d}] out of [{3:d}, {4:d}].'.\
                     format(name, i, j, dF.shape[0], dF.shape[1]))
            cg.write(45 * '#')

            # store positivity information for later lookup
            positivity_information = np.ones(self.diffOrder + 2).astype(bool)
            positivity_information_nonlocal = np.zeros(self.diffOrder).astype(bool)

            order = 0
            while order <= diffOrder:
                # Only call simplify when we have no non-local problems ->
                #   messes up integral derivative ordering.
                if not self.isNonLocal:
                    ex = simplify(exp.coeff(syp.Derivative(pfun, (x, order))))
                else:
                    ex = exp.coeff(syp.Derivative(pfun, (x, order)))

                # these pieces of code need to be callable
                # x, u, v -> to lambda x: -> for Fun computation
                function_names = []
                function_names += self.function_names
                if self.isNonLocal: function_names += self.integration_kernels
                ccode = cpcode(ex, function_names=function_names,
                               allow_unknown_functions=True, no_evaluation=True)

                # Write to the code gen
                cg.write('def {0:s}{1:d}{2:d}{3:d}({4}):'.format(name, i, j, order,
                                                                 ', '.join(map(str, self.function_names))))
                cg.indent()

                # Depending on the type of expression we need to write the numpy code differently!
                real = ex.is_real if ex.is_real is not None else False
                if ex.is_zero:
                    cg.write('return zeros(1, domain=[{0:.16f}, {1:.16f}], type=\'{2:s}\')'.format(self.domain[0], self.domain[1], self.ftype))
                    positivity_information[order] = False
                elif ex.is_constant() or real:
                    cg.write('return {0} * ones(1, domain=[{1:.16f}, {2:.16f}], type=\'{3:s}\')'.format(ccode, self.domain[0], self.domain[1], self.ftype))
                else:
                    cg.write('return {0}'.format(ccode))

                cg.write('')
                cg.dedent()
                order += 1

            # Check whether we have any convolution terms
            if self.isNonLocal:
                conv_order = 0
                kernel = self.kernels[0]
                exprKp = convolve(pfun,  kernel, x, lower_limit=-1, upper_limit=1)
                while conv_order <= self.diffOrder - 1:
                    # Only call simplify when we have no non-local problems ->
                    #   messes up integral derivative ordering.
                    ex = exp.coeff(syp.Derivative(exprKp, (x, conv_order)))

                    # these pieces of code need to be callable
                    # x, u, v -> to lambda x: -> for Fun computation
                    function_names = []
                    function_names += self.function_names
                    if self.isNonLocal: function_names += self.integration_kernels
                    ccode = cpcode(ex, function_names=function_names,
                                   allow_unknown_functions=True,
                                   no_evaluation=True)

                    # Write to the code gen
                    cg.write('def {0:s}{1:d}{2:d}{3:d}({4}):'
                             .format(nonlocal_name, i, j, conv_order,
                                     ', '.join(map(str, self.function_names))))
                    cg.indent()

                    # Depending on the type of expression we need to write the numpy code differently!
                    real = ex.is_real if ex.is_real is not None else False
                    if ex.is_zero:
                        cg.write('return zeros(1, domain=[{0:.16f}, {1:.16f}], type=\'{2:s}\')'.format(self.domain[0], self.domain[1], self.ftype))
                    elif ex.is_constant() or real:
                        cg.write('return {0} * ones(1, domain=[{1:.16f}, {2:.16f}], type=\'{3:s}\')'.format(ccode, self.domain[0], self.domain[1], self.ftype))
                        positivity_information_nonlocal[conv_order] = True
                    else:
                        positivity_information_nonlocal[conv_order] = True
                        cg.write('return {0}'.format(ccode))

                    cg.write('')
                    cg.dedent()
                    conv_order += 1

            # Check whether we have any integral terms!
            ex = simplify(exp.coeff(syp.Integral(pfun)))
            cg.write('def {0:s}{1:d}{2:d}{3:s}({4}):'.format(name, i, j, 'i',
                                                             ', '.join(map(str, self.function_names))))
            cg.indent()

            if not ex.is_zero:
                real = ex.is_real if ex.is_real is not None else False
                # Compile the code into something useful
                ccode = cpcode(ex, function_names=self.function_names,
                               allow_unknown_functions=True, no_evaluation=True)
                if ex.is_constant() or real:
                    cg.write('return {0} * ones(1, domain=[{1:.16f}, {2:.16f}], type=\'{3:s}\')'.format(ccode, self.domain[0], self.domain[1], self.ftype))
                else:
                    cg.write('return {0}'.format(ccode))
            else:  # nothing -> write zero
                positivity_information[-1] = False
                cg.write('return zeros(1, domain=[{0:.16f}, {1:.16f}], type=\'{2:s}\')'.format(self.domain[0], self.domain[1], self.ftype))
            cg.dedent()

            # create code for positivity information
            cg.write('')
            cg.write('# Return sorted by derivative term order, followed by the integral terms.')
            cg.write('def {0:s}{1:s}{2:d}{3:d}():'.format(name, 'info', i, j))
            cg.indent()
            cg.write('return [{0}]'.format(', '.join(map(str, positivity_information))))
            cg.dedent()

            # create code for positivity information
            cg.write('')
            cg.write('# Return sorted by derivative term order, followed by the integral terms.')
            cg.write('def {0:s}{1:s}{2:d}{3:d}():'.format(nonlocal_name, 'info', i, j))
            cg.indent()
            cg.write('return [{0}]'.format(', '.join(map(str, positivity_information_nonlocal))))
            cg.dedent()

            # get the code block
            matrix[i, j] = cg.end()
            if debug: print(matrix[i, j])

        return matrix

    def init(self, debug=False):
        # generate the Sympy base program
        pycode = sympy_base_program(self.function_names, self.operator_names,
                                    self.integration_kernels, self.parameter_names,
                                    constant_functions=self.constant_functions)
        execute_pycode(pycode, self.ns, debug=debug)

    def sympify(self, src, obj_name, debug=False):
        # create the Matrix for the nonlinear operator
        n_src = len(src)
        matrix = zeros(n_src, 1)
        for i, j in np.ndindex(matrix.shape):
            matrix[i, j] = sympify(src[i], locals=self.ns)
        return matrix

    def eqn_parameter_derivatives(self, src, p_name, debug=False):
        assert p_name in self.parameter_names.keys(), '%s not a known parameter!' % p_name
        p = self.ns[p_name]
        self.dmdp[p_name] = syp.diff(src, p)
        # TODO: fix correctly!
        pycode = self.__emit_dp({p_name: self.dmdp[p_name]})
        return pycode

    def __emit_common(self, var_names=['x']):
        """ Generates the code to have a spatial variables such as x available to the code """
        cg = CodeGeneratorBackend()
        cg.begin(tab=4*" ")
        cg.write('#!/usr/bin/python')
        cg.write('# -*- coding: utf-8 -*-')
        cg.write(35 * '#')
        cg.write('# This is the COLLOC BASE PROGRAM #')
        cg.write('#                                 #')
        cg.write('#     I am auto-generated!        #')
        cg.write('#     Do not mess with me!        #')
        cg.write('#                                 #')
        cg.write(35 * '#')
        cg.write('# author: Andreas Buttenschoen {0}'.format(datetime.datetime.now().year))
        cg.write('import numpy')
        cg.write('import scipy')
        cg.write('import numpy as np')
        cg.write('import funpy as fp')
        cg.write('from funpy import *')
        cg.write('')
        cg.write(35 * '#')
        cg.write('# Spatial variables')
        cg.write(35 * '#')
        cg.write('domain=np.asarray([{0:.16f}, {1:.16f}])'.format(self.domain[0], self.domain[1]))

        for var in var_names:
            cg.write('{0} = Fun(op=lambda x: x, domain=[{1:.16f}, {2:.16f}])'.format(var, self.domain[0], self.domain[1]))

        if self.isNonLocal:
            cg.write('')
            cg.write(35 * '#')
            cg.write('# Non-local integration kernels')
            cg.write(35 * '#')
            cg.write('from colloc.kernels import uniform_kernel')
            cg.write('k = uniform_kernel(1024, domain)')

        #cg.write('_cos_ = Fun(op=lambda x: np.cos(x), type=\'{0:s}\', domain=[{1:.16f}, {2:.16f}])'.format(self.ftype, self.domain[0], self.domain[1]))
        #cg.write('_sin_ = Fun(op=lambda x: np.sin(x), type=\'{0:s}\', domain=[{1:.16f}, {2:.16f}])'.format(self.ftype, self.domain[0], self.domain[1]))
        cg.write('')

        # Create simple functions to debug parameter values etc.
        cg.write(35 * '#')
        cg.write('# Debug code')
        cg.write(35 * '#')
        cg.write('def print_pars():')
        cg.indent()

        cg.write("rstr=''")
        for p_name, p_value in self.parameter_names.items():
            cg.write('rstr += \"{0:s} = %.4g, \" % {0:s}'.format(p_name))
        cg.write('return rstr')
        cg.dedent()
        return cg.end()

    def __emit_dp(self, src):
        cg = CodeGeneratorBackend()
        cg.begin(tab=4*" ")

        for p_name in self.parameter_names.keys():
            # Skip names that we don't need yet
            if not p_name in src.keys():
                continue

            for i in range(src[p_name].shape[0]):
                function_names = []
                function_names += self.function_names
                if self.isNonLocal:
                    kOp = self.operators[0]   # Assuming only one operator for the moment
                    kernel = self.kernels[0]
                    x = self.ns['x']
                    exprK = convolve(self.functions[0],  kernel, x, lower_limit=-1, upper_limit=1)
                    dp = src[p_name][i, 0].replace(kOp, exprK)
                    if self.isNonLocal: function_names += self.integration_kernels
                else:
                    dp = simplify(src[p_name][i, 0])

                fcode = cpcode(dp, function_names=function_names,
                               allow_unknown_functions=True, no_evaluation=True)

                cg.write('def dp_{0:s}_{1:s}({2}):'.format(p_name, self.function_names[i],
                                                           ', '.join(map(str, self.function_names))))
                cg.indent()
                if dp.is_constant():
                    cg.write('return {0} * np.ones_like({1})'.format(fcode, self.function_names[i]))
                else:
                    cg.write('return {}'.format(fcode))

                cg.dedent()
                cg.write('')

        return cg.end()

    def __emit_dpdp(self, src, p_name, func_name='dpdp'):
        cg = CodeGeneratorBackend()
        cg.begin(tab=4*" ")

        # Here the function names are those of the first derivative!
        # Inputs are -> (u, v, ..., up, vp, ... , upp, vpp, .... )
        # The resulting matrix is applied to the representation of upp, vpp, ....
        function_names = []
        function_names += self.function_names
        function_names += self.pfunc_names

        for i in range(src.shape[0] - 1):
            dp = src[i, 0]

            fcode = cpcode(dp, function_names=function_names,
                           allow_unknown_functions=True, no_evaluation=True)

            cg.write('def {0:s}_{1:s}_{2:s}({3}):'.format(func_name, p_name, self.function_names[i],
                                                         ', '.join(map(str, function_names))))
            cg.indent()
            if dp.is_constant():
                cg.write('return {0} * np.ones_like({1})'.format(fcode, self.function_names[i]))
            else:
                cg.write('return {}'.format(fcode))

            cg.dedent()
            cg.write('')

        # Also write function for the last element
        dp = src[-1, 0]
        fcode = cpcode(dp, function_names=function_names,
                       allow_unknown_functions=True, no_evaluation=True)

        cg.write('def {0:s}_{1:s}_{2:s}({3}):'.format(func_name, p_name, p_name,
                                                     ', '.join(map(str, function_names))))
        cg.indent()
        if dp.is_constant():
            cg.write('return {0}'.format(fcode))
        else:
            cg.write('return {}'.format(fcode))

        cg.dedent()
        cg.write('')
        return cg.end()

    def __emit_dxdp(self, src, p_name, func_name='dxdp'):
        cg = CodeGeneratorBackend()
        cg.begin(tab=4*" ")

        # Here the function names are those of the first derivative!
        # Inputs are -> (u, v, ..., up, vp, ... , upp, vpp, .... )
        # The resulting matrix is applied to the representation of upp, vpp, ....
        function_names = []
        function_names += self.function_names
        function_names += self.pfunc_names

        # Define the perturbation parameter name
        p_symbol = self.par_pert[p_name]

        for i in range(src.shape[0]):
            dp = simplify(src[i, 0] / p_symbol)

            fcode = cpcode(dp, function_names=function_names,
                           allow_unknown_functions=True, no_evaluation=True)

            cg.write('def {0:s}_{1:s}_{2:s}({3}):'.format(func_name, p_name, self.function_names[i],
                                                         ', '.join(map(str, function_names))))
            cg.indent()
            if dp.is_constant():
                cg.write('return {0} * np.ones_like({1})'.format(fcode, self.function_names[i]))
            else:
                cg.write('return {}'.format(fcode))

            cg.dedent()
            cg.write('')

        return cg.end()

    def __emit_function(self, src, name='f', backend=cpcode, debug=False):
        cg = CodeGeneratorBackend()
        cg.begin(tab=4*" ")
        cg.write(35 * '#')
        cg.write('# Function block {0:s}.'.format(name))
        cg.write(35 * '#')

        for i in range(src.shape[0]):
            # Transform sympy into a callable numpy function
            if self.isNonLocal:
                # if non-local need to create expressions of the non-local terms!
                ops  = self.operators
                kOp = ops[0]   # Assuming only one operator for the moment
                kernel = self.kernels[0]
                x = self.ns['x']
                exprK = convolve(self.functions[0],  kernel, x, lower_limit=-1, upper_limit=1)
                func = backend(src[i, 0].replace(kOp, exprK), no_evaluation=True)
            else:
                func = backend(src[i, 0], no_evaluation=True)

            cg.write('def {0:s}{1:d}({2}):'.format(name, i, ', '.join(map(str, self.function_names))))
            cg.indent()
            cg.write('return asfun({0}, type=\'{1}\', domain=[{2:.16f}, {3:.16f}])'
                     .format(func, self.ftype, self.domain[0], self.domain[1]))
            cg.dedent()
            cg.write('')

        return cg.end()
