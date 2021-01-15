#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import datetime
from sympy import oo, integrate, Symbol
from sympy.core.compatibility import exec_
from ac.gen import CodeGeneratorBackend


def execute_pycode(pycode, namespace, debug=False):
    """ Function executes code into a given namespace """
    if debug:
        print(pycode)

    # Execute the generate code
    try:
        exec_(pycode, namespace)
    except Exception as e:
        raise e

def sympy_base_program_ode(function_names, parameters={}):
    cg = CodeGeneratorBackend()
    cg.begin(tab=4*" ")
    cg.write('#!/usr/bin/python')
    cg.write('# -*- coding: utf-8 -*-')
    cg.write('# author: Andreas Buttenschoen {0}'.format(datetime.datetime.now().year))
    cg.write('# Do not modify! File auto generated!')
    cg.write('import numpy')
    cg.write('import numpy as np')
    cg.write('')

    cg.write('from sympy import symbols, Function, integrate, Derivative')
    cg.write('from sympy import Derivative as diff')
    cg.write('from sympy import integrate as sum')
    cg.write('from sympy import integrate as int')
    cg.write('x, y, z = symbols("x y z", real=True)')

    for function_name in function_names:
        cg.write('{0} = symbols("{0}", real=True, constant=True)'.format(function_name))

    # Create symbols for the parameters
    for p_name, p_value in parameters.items():
        cg.write('{0} = symbols("{0}", real=True, positive=True)'.format(p_name))

    # Create simple functions to debug parameter values etc.
    cg.write('')
    cg.write('def print_pars():')
    cg.indent()

    cg.write("rstr=''")
    for p_name, p_value in parameters.items():
        cg.write('rstr += \"{0:s} = %.4g, \" % {0:s}'.format(p_name))
    cg.write('return rstr')
    cg.dedent()
    return cg.end()

def sympy_base_program(function_names, operator_names=[],
                       kernels=[], parameters={},
                       unlikely_var_name='dummy_variable',
                       constant_functions=False, positive_parameters=True,
                       real=True):
    cg = CodeGeneratorBackend()
    cg.begin(tab=4*" ")
    cg.write('#!/usr/bin/python')
    cg.write('# -*- coding: utf-8 -*-')
    cg.write('# author: Andreas Buttenschoen {0}'.format(datetime.datetime.now().year))
    cg.write(35 * '#')
    cg.write('# This is the SYMPY BASE PROGRAM. #')
    cg.write('#                                 #')
    cg.write('#     I am auto-generated!        #')
    cg.write('#     Do not mess with me!        #')
    cg.write('#                                 #')
    cg.write(35 * '#')
    cg.write('import numpy')
    cg.write('import numpy as np')
    cg.write('')

    cg.write('from sympy import symbols, Function, integrate, Derivative')
    cg.write('from sympy import Symbol, oo')
    cg.write('from sympy import Derivative as diff')
    cg.write('from sympy import integrate as sum')
    cg.write('from sympy import integrate as int')
    cg.write('from sympy.functions import sign')
    cg.write('')
    cg.write('# Domain symbols')
    cg.write('x, y, z = symbols("x y z", real=True)')
    cg.write('')

    # Convolution support?
    # cg.write('def convolve(f, g, t, lower_limit=-oo, upper_limit=oo):')
    # cg.indent()
    # cg.write('tau = Symbol(\'{0:s}\', real=True)'.format(unlikely_var_name))
    # cg.write('return integrate(f.subs(t, tau) * g.subs(t, t - tau), (tau, lower_limit, upper_limit))')
    # cg.dedent()

    # TODO: Why doesn't sympy simplify constant functions ??
    cg.write('# Function symbols')
    for function_name in function_names:
        if constant_functions:
            cg.write('{0} = symbols("{0}", real={1:}, constant={2:})'.format(function_name, real, constant_functions))
        else:
            cg.write('{0} = Function("{0}", real={1:}, constant={2:})(x)'.format(function_name, real, constant_functions))

    # Define operators
    if operator_names:
        cg.write('')
        cg.write('# Operator names')
        for operator_name in operator_names:
            cg.write('{0} = Function("{0}", real={1:})(x)'.format(operator_name, real))

    # Define integral kernels!
    if kernels:
        cg.write('')
        cg.write('# Integration kernels names')
        for kernel in kernels:
            cg.write('{0} = Function("{0}", real={1:})(x)'.format(kernel, real))

    # Create symbols for the parameters
    cg.write('')
    cg.write(35 * '#')
    cg.write('# Parameter Block                 #')
    cg.write(35 * '#')
    cg.write('# Equation parameters')
    for p_name, p_value in parameters.items():
        cg.write('{0} = symbols("{0}", real=True, positive=True)'.format(p_name))

    # Create simple functions to debug parameter values etc.
    cg.write('')
    cg.write('def print_pars():')
    cg.indent()

    cg.write("rstr=''")
    for p_name, p_value in parameters.items():
        cg.write('rstr += \"{0:s} = %.4g, \" % {0:s}'.format(p_name))
    cg.write('return rstr')
    cg.dedent()
    return cg.end()

def sympy_rhs_program(eqns):
    cg = CodeGeneratorBackend()
    cg.begin(tab=4*" ")
    cg.write('#!/usr/bin/python')
    cg.write('# -*- coding: utf-8 -*-')
    cg.write('# author: Andreas Buttenschoen {0}'.format(datetime.datetime.now().year))
    cg.write('# Do not modify! File auto generated!')
    cg.write('from sympy import symbols, Function, Derivative')
    cg.write('from sympy.matrices import Matrix')
    cg.write('from sympy import sympify')
    cg.write('')

    matrix_str = 'Matrix(['
    m_max = len(eqns) - 1
    for i, eqn in enumerate(eqns):
        matrix_str += '[sympify({0})]'.format(eqn)
        if i < m_max:
            matrix_str += ', '
    matrix_str += '])'

    cg.write('eqn = {0}'.format(matrix_str))
    return cg.end()

def pycode_imports():
    cg = CodeGeneratorBackend()
    cg.begin(tab=4*" ")
    cg.write('#!/usr/bin/python')
    cg.write('# -*- coding: utf-8 -*-')
    cg.write('# author: Andreas Buttenschoen {0}'.format(datetime.datetime.now().year))
    cg.write('# Do not modify! File auto generated!')
    cg.write('import numpy')
    cg.write('import scipy')
    cg.write('import numpy as np')
    cg.write('from fun import *')
    cg.write('')
    return cg.end()

def convolve(f, g, t, dummy_var_name='r', lower_limit=-oo, upper_limit=oo):
    """ Generates the expression for non-local adhesion operators
        -> not really a convolution, but very similar.
    """
    dummy_int_var = Symbol(dummy_var_name, real=True)
    return integrate(f.subs(t, dummy_int_var + t) * g.subs(t, dummy_int_var),
                     (dummy_int_var, lower_limit, upper_limit))
