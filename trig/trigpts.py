#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np

def quadwts(n):
    """ Quadrature weights for equally spaced points from [-1, 1) """
    return 2/n * np.ones(n)

def trigpts(n):
    """ Equispaced points in [-1, 1] """
    if n <= 0:
        x = np.zeros(0)
        w = np.zeros(0)
        return x, w

    x = np.linspace(-1, 1, n+1)[:-1]
    w = quadwts(n)
    return x, w
