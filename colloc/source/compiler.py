#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import itertools
import numpy as np
import sympy as syp
import datetime

from sympy import symbols, simplify, Function, Integral
from sympy.matrices import zeros, ones
from sympy import sympify, latex, pprint, preorder_traversal
from IPython.display import display

from ac.support import Namespace
from ac.gen import CodeGeneratorBackend

from cheb.cbcode import cpcode
from cheb.odecode import odecode
from colloc.tools import execute_pycode, sympy_base_program, convolve
from colloc.source.OperatorSource import OperatorSource
from colloc.source.NonlinearOperatorSource import NonlinearFunctionSource
from colloc.source.NonlinearVectorFunctionSource import NonlinearVectorFunctionSource
from colloc.source.DiffOperatorSource import DiffOperatorSource
from colloc.source.IntegralOperatorSource import IntegralOperatorSource
from colloc.source.function import DummyFunction


class Source:
    """ Source for the discretization of an operator on a Hilbert space.

        The required Frechet derivatives are computed symbolically using SymPy.

        TODO: Sometime in the future improve the quality and modularization of
        the code below. Right now this is completely horrific.
    """
    def __init__(self, eqn_src, cts_src, function_names,
                 constant_function_names, operator_names, parameter_names,
                 integration_kernels, *args, **kwargs):
        self.ns = Namespace()

        self.diffOrder = kwargs.pop('diffOrder', 2)

        self.eqn_src = eqn_src
        self.cts_src = cts_src
        self.ftype = kwargs.pop('ftype', 'cheb')

        self.function_names = function_names
        self.constant_function_names = constant_function_names
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
                             'bif': 'moore', 'fold': 'fold', 'cts_fold': 'cts_fold',
                             'rhs_fold': 'rhs_fold', 'rhs_curv': 'rhs_curv'}

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
        self.dp   = {}          # Compiled functions for the dp
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
        self.pfuns = {}

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

    @property
    def n_eqn(self):
        return len(self.eqn_src)

    @property
    def n_cts(self):
        return len(self.cts_src)

    def get_function_names(self, type=None, constant=False):
        function_names = []
        function_names += self.function_names

        if constant:
            function_names += self.constant_function_names

        if type is None:
            return function_names
        elif type == 'bilin':
            function_names += [f.name for f in self.pfuns['p']]
        elif type == 'trilin':
            function_names += [f.name for f in self.pfuns['pp']]

        return function_names

    def compile(self, cpars=[], debug=False, par=False, fold=False, bif=False):
        self.init(debug=debug)
        self.functions = [self.ns[fname] for fname in self.function_names]
        self.operators = [self.ns[oname] for oname in self.operator_names]
        self.kernels   = [self.ns[kname] for kname in self.integration_kernels]

        # store the parameter names for which compiled stuff
        if not isinstance(cpars, list):
            cpars = [cpars]
        self.compiled_pars += cpars

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

            # Create a function source object representing the residual!
            func = NonlinearVectorFunctionSource(spcode, self.ftype,
                                                 self.domain, name=name,
                                                 functions=self.function_names)
            # FIXME
            func.ftype = self.ftype
            func.domain = self.domain
            setattr(self, type, func)

            # Generate linearization
            dname = 'd' + type
            d_spcode = self.linearize(spcode, self.diffOrder, name=name, debug=debug)
            d_spcode.name = type  # IMPROVE!
            # FIXME
            self.d_spcode[type] = d_spcode
            self.syp_src[dname] = d_spcode

        # Compile the functions computing the first derivatives with respect
        # to the defined parameters!
        if par:
            function_names = self.get_function_names()
            self.compile_dp(cpars, function_names, debug=debug)

        if fold:
            function_names = self.get_function_names(type='bilin')
            self.compile_fold(cpars, function_names, debug=debug)

        if bif:
            assert False, 'Fix me once again!'

    def compile_fold(self, cpars, function_names, debug=False):
        for type in ['eqn', 'cts']:
            # Get the action of the linearization
            daction = self.syp_src['d' + type].action
            self.syp_src['ddd' + type] = daction

            name = 'd' + self.symbol_names[type]

            # Create a function source object representing the residual!
            func = NonlinearVectorFunctionSource(daction, self.ftype,
                                                 self.domain, name=name,
                                                 functions=function_names)
            if debug: print(func.emit())

            # TODO give me the right names
            # setattr(self, type, func)
            self.syp_src['dd' + type] = self.linearize(daction, self.diffOrder,
                                                       order=2,
                                                       name=name, debug=debug)

        # Compute a bunch of coefficients
        self.compile_dp(cpars, function_names, debug=debug, name='dxdp', src_name='dddeqn')

    def compile_dp(self, cpars, function_names, debug=False, name='dp', src_name='eqn'):
        src = self.syp_src[src_name]
        setattr(self, name, {})
        dest = getattr(self, name)
        for cpar in cpars:
            assert cpar in self.parameter_names.keys(), '{} not a known parameter!'.format(cpar)

            # Look up parameter symbol and differentiation + simplify
            p = self.ns[cpar]
            p_src = simplify(syp.diff(src, p), ratio=1)

            # Create output function for derivatives!
            func = NonlinearVectorFunctionSource(p_src, self.ftype, self.domain,
                                                 name='{0:s}_{1:s}'.format(name, cpar),
                                                 functions=function_names)

            # Store this function
            if debug: print(func.emit())
            dest[cpar] = func

    def get_pfuns(self, order):
        key = order * 'p'
        if key not in self.pfuns.keys():
            x = self.ns['x']
            self.pfuns[key] = [DummyFunction(name + '_' + key, x) for name in self.function_names]

        return self.pfuns[key]

    def linearize(self, matrix, diffOrder, order=1, name='f', debug=False, simplify=False):
        """ Computes the linearization of the operator stored in matrix """
        # replace each of the functions with the perturbation
        x = self.ns['x']
        pfuns = self.get_pfuns(order)

        # The increment used to compute the Frechet derivative
        e = syp.symbols('eps_' + order * 'p', real=True)

        # Need to group values
        n = matrix.shape[0]
        m = len(self.functions)
        DM = zeros(n, m)

        # look up the sympy objects
        funs = self.functions
        ops  = self.operators
        assert len(ops) <= 1, 'Can only handle at most one operator!'

        for i, j in np.ndindex(DM.shape):
            fun, pfun = funs[j], pfuns[j].expr
            exp = matrix[i, 0].replace(fun, fun + e * pfun)

            # Compute the partial Frechet derivative!
            # Make sure to call expand here -> otherwise subsequent coeff might fail!
            DM[i, j] = syp.collect(syp.diff(exp, e).subs({e: 0}), pfun)

        # Create the operator source representation
        osrc = OperatorSource(DM.shape, func_names=self.function_names,
                              cfunc_names=list(self.constant_function_names.keys()),
                              pars=self.parameter_names)

        for i, j in np.ndindex(DM.shape):
            pfun = pfuns[j].expr

            ex = syp.simplify(DM[i, j].coeff(pfun), ratio=1)
            op = DiffOperatorSource(fun=pfun, coeffs=ex, dummy=x, order=0)
            osrc[i, j].append(op)

            order = 1
            while order <= diffOrder:
                dpfun = syp.Derivative(pfun, (x, order))
                ex = syp.simplify(DM[i, j].coeff(dpfun), ratio=1)
                op = DiffOperatorSource(fun=pfun, coeffs=ex, dummy=x, order=order)
                osrc[i, j].append(op)
                order += 1

            # Expand now -> otherwise trouble getting non-local coefficients
            DM[i, j] = syp.expand(DM[i, j])

            # Collect integral terms so that we can extract them!
            integrals = [arg for arg in preorder_traversal(DM[i, j]) if isinstance(arg, Integral)]
            for dpfun in integrals:
                # Extract the coefficient
                ex = syp.simplify(DM[i, j].coeff(dpfun), ratio=1)

                # Extract the integrand
                integrand = syp.simplify(dpfun.function.coeff(pfun), ratio=1)

                op = IntegralOperatorSource(coeffs=ex, fun=pfun,
                                            integrand=integrand, dummy=x)
                osrc[i, j].append(op)

        # Setup the source
        osrc.finish()

        # Generate code to compute the coefficients of the linear operator
        # function_names = self.get_function_names(constant=True)
        #for i, j in np.ndindex(osrc.shape):
        #    block = osrc[i, j]

        #    # TEMP HACK
        #    # block.domain = self.domain
        #    # block.ftype = self.ftype
        #    # block.function_names = function_names
        #    # block.name = name

        return osrc

    def init(self, debug=False):
        # generate the Sympy base program
        pycode = sympy_base_program(self.function_names, self.constant_function_names,
                                    self.operator_names,
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
        cg.write('from fun import *')
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

        cg.write('# Constant Function symbols')
        for function_name, function_code in self.constant_function_names.items():
            cg.write('{0} = Fun(op={1}, domain=[{2:.16f}, {3:.16f}])'.\
                     format(function_name, function_code,
                             self.domain[0], self.domain[1]))

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
