#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import datetime
import numpy as np
from ac.gen import CodeGeneratorBackend


def execute_pycode(pycode, namespace, debug=False):
    """ Function executes code into a given namespace """
    if debug: print(pycode)

    # Execute the generate code
    try:
        exec(pycode, namespace)
    except Exception as e:
        raise e


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


def remove_bc(iar, n_eqn, bw=2):
    n = iar.size // n_eqn
    m = n_eqn * (n - bw)

    out = np.empty(m, dtype=iar.dtype)

    for i in range(n_eqn):
        for j in range(n - bw):
            out[i * (n - bw) + j] = iar[i * n + j + bw]

    return out.reshape((n - bw, n_eqn), order='F')


def replace_bc(iar, n_eqn, bw=2, reshape=False):
    n = iar.size // n_eqn

    out = np.zeros_like(iar)

    for i in range(n_eqn):
        for j in range(n - bw):
            out[i * n + j] = iar[i * n + j + bw]

    if reshape:
        return out.reshape((n, n_eqn), order='F')
    else:
        return out


def get_bc(iar, n_eqn, bw=2):
    n = iar.size // n_eqn
    out = np.zeros(n_eqn * bw, dtype=float)

    for i in range(n_eqn):
        for j in range(bw):
            out[i * bw + j] = iar[i * n + j]

    return out
