#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
import warnings

from fun import Fun


def uniform_kernel(n, domain):
    # Non-local operator essentially multiplies all the coefficients by
    #
    # sin(k) / k  when k not 0
    #
    # and sets it to zero when k = 0

    def gen_m0(i, Rscl):
        return 1j * np.piecewise(i, [i == 0, (i < 0) | (i > 0)],
                                 [lambda k: 0, lambda k: (1.0 - np.cos(Rscl * k)) / (Rscl * k)])

    rescaleFactor = 0.5 * np.diff(domain)
    Rscl = np.pi / rescaleFactor

    # Silence warnings / exceptions that we will encounter next! But don't worry we will fix it!
    if np.remainder(n, 2):  # n is odd
        k = np.arange(-(n-1)/2, n/2, 1)
    else:  # n is even
        k = np.arange(-n/2, n/2, 1)

    even_odd_fix = gen_m0(k, Rscl)
    return Fun(coeffs=np.expand_dims(even_odd_fix, axis=1), domain=domain, type='trig')
