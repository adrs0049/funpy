# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
#cython: language_level=3
#cython: annotate=True
#cython: infer_types=True
import numpy as np
cimport numpy as np
cimport cython

cpdef polyfit(const double[:, :] sampled, int workers = *)
cpdef polyval(const double[:, :] chebcoeff, int workers = *)
cpdef prolong(double[::1, :] array, int Nout)
cpdef simplify_coeffs(double[::1, :] coeffs, eps = *)
