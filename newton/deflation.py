#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np

from fun import h1norm
from support.Functional import Functional


def deflation(u, known_solutions=[], shift=1.0, power=2.0):
    return 1.0 if not known_solutions else np.product([shift + 1.0 / h1norm(u - ks)**power for ks in known_solutions])


def deflation_linearization(u, known_solutions=[], shift=1.0, power=2.0, basis='coeff'):
    """ Computes the derivative of the deflation operator eta(u)

            eta'(u) = 2 eta(u) Sum [ u - u_k / [ shift ( | u - u_k |^2 + 1 ) | u - u_k |^2 ] ]
                                k

            This needs to be implemented as a LinearOperator
    """
    eta = 1.
    if not known_solutions:
        return eta, None

    # Compute the function that defines the functional
    function = np.zeros_like(u)
    for ks in known_solutions:
        diff = u - ks
        diff_norm = h1norm(diff)

        # deflation factor
        defl = shift + 1. / diff_norm**power
        eta *= defl

        # scale the function appropriately
        function += diff / (defl * diff_norm**(2 + power))

    # multiply by the common factor!
    function *= -power * eta

    # Make sure that function has the correct shape
    if abs(function.shape[0] - u.shape[0]) > 0:
        function.prolong(u.shape[0])

    # create Functional representing the derivative of eta
    return eta, Functional(function, order=1, basis=basis)
