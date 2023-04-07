#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
from math import sqrt
import scipy.linalg as LA


class DenseLUSolver:
    def __init__(self, A, *args, **kwargs):
        self.lu, self.piv = LA.lu_factor(A)

    def solve(self, b):
        return LA.lu_solve((self.lu, self.piv), b)
