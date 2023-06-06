#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
import scipy.sparse as sps
import scipy.linalg as LA
import scipy.sparse.linalg as LAS

from .qr_chol import QRCholesky
from .lu_solve import DenseLUSolver


class ExactLinearSolver:
    def __init__(self, A, proj=lambda x: x, to_fun=lambda x: x, *args, **kwargs):
        # Store
        self.proj = proj
        self.m, self.n = A.shape
        self.to_fun = to_fun

        # Is the matrix sparse?
        if isinstance(A, LAS.LinearOperator): A = A.to_matrix()
        sparse = sps.issparse(A)

        # What linear method are we using?
        method = kwargs.pop('method', 'lu').lower()

        if method == 'lu' and sparse:
            self.solver = LAS.splu(A.tocsc())
        elif method == 'lu' and not sparse:
            self.solver = DenseLUSolver(A, *args, **kwargs)
        elif method == 'qr':
            self.solver = QRCholesky(A, rank=self.A.shape[0], *args, **kwargs)
        else:
            raise RuntimeError(f'Unknown exact linear solver {method} requested!')

    def solve(self, b):
        # Solve the linear system
        x = self.solver.solve(b)

        # Map the result back to the correct function object
        x = self.to_fun(self.proj(x))

        return x, True
