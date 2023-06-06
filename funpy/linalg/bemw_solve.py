#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
import scipy.sparse as sps


class BEMW:
    """
        Solve the matrix

            / A c \  / x \  =  / f \
            |     |  |   |  =  |   |
            \ b d /  \ z /  =  \ h /

        using the mixed block elimination method BEM.

        For the matrix above to be nonsingular we need that:
            1. C spans a subspace transversal to R[A] i.e. N[A^T]
            2. B spans a subspace transversal to R[A^T] i.e. N[A]

        To construct the matrix with the smallest condition number we set:
            1. d = 0
            2. B, C, to be orthonormal sets of the above basis.

    """
    def __init__(self, solver, bs, cs, ds):
        assert len(bs) == len(cs) == len(ds), ''

        self.m = len(bs)
        self.n = solver.shape[0]

        # Main storage
        self.solver = solver
        self.bs = bs
        self.cs = cs
        self.ds = ds

        # Storage for data computed by this class
        self.storage = None

        # If not singular do the pre-processing steps for BEM.
        if not self.is_singular: self.prepare()

    def det(self):
        # FIXME
        return self.solver.det() * (self.d - np.dot(self.b, self.beta))

    def prepare(self):
        """
            Storage memory layout for the required data:

            --------------------------------
            |  1 |  1 |  n + m - 1  |  n   |
            -------------------------------|
            | δ1 | ε1 |      w1     |  vm  |
            | .  | .  |       .     |  .   |
            | .  | .  |       .     |  .   |
            | δm | εm |      wn     |  v1  |
            --------------------------------

            Note that the third boundary moves as the sizes of vj and wj change
            as going down the rows. The total memory usage is: m x (1 + 2n + m)

        """
        n = self.n
        m = self.m

        # global storage
        b = self.bs
        c = self.cs
        d = self.ds

        # Create factor storage
        self.storage = np.empty((m, m + 2 * n + 1), dtype=float)

        for k in range(m):
            # vk: v1 has length n
            vk = self.solve_detail(k, c[k])
            r = m - k - 1
            self.storage[r, 2 + r + n:] = vk
            self.storage[k, 0] = d[k] - np.dot(b[k], vk)

            # wk: w1 has length n + m - 1
            wk = self.solve_adj_detail(k, b[k])
            self.storage[k, 2:n + k + 2] = wk
            self.storage[k, 1] = d[k] - np.dot(c[k], wk)

    def update_row(self, j, b, d):
        assert j >= 0 and j < self.m, ''

        self.bs[j] = b
        self.ds[j] = d

        # Update the pre-processing steps
        # TODO: optimize!
        if not self.is_singular: self.prepare()

    def update_col(self, j, c, d):
        assert j >= 0 and j < self.m, ''

        self.cs[j] = c
        self.ds[j] = d

        # Update the pre-processing steps
        # TODO: optimize!
        if not self.is_singular: self.prepare()

    """ Simple lookup functions of the values in storage """
    def δ(self, k):
        return self.storage[k, 0]

    def ε(self, k):
        return self.storage[k, 1]

    def v(self, k):
        r = self.m - k - 1
        return self.storage[r, 2 + r + self.n:]

    def w(self, k):
        return self.storage[k, 2:self.n + k + 2]

    def solve_detail(self, j, x, overwrite=False):
        n = self.n

        # global storage
        b = self.bs
        c = self.cs
        d = self.ds

        # local storage
        y = np.empty(j, dtype=float)

        if not overwrite:
            x = x.copy()

        # Type casting!
        x = x.astype(float)

        # Step 1:
        for i in range(j, 0, -1):
            k = i - 1

            # Step 2:
            y[k] = (x[n + k] - np.dot(self.w(k), x[:n + k])) / self.ε(k)

            # Step 3:
            x[:n + k] -= c[k] * y[k]
            x[n + k]  -= d[k] * y[k]

        # Step 4: Solve core problem
        x[:n] = self.solver.solve(x[:n])

        # Step 5:
        for k in range(0, j):
            # Step 6: test
            ypp       = (x[n + k] - np.dot(b[k], x[:n + k])) / self.δ(k)
            x[n + k]  = y[k]
            x[:n + k] -= self.v(k) * ypp
            x[n + k]  += ypp

        return x

    def solve_adj_detail(self, j, x, overwrite=False):
        n = self.n

        # global storage
        b = self.bs
        c = self.cs
        d = self.ds

        # local storage
        y = np.empty(j + 1, dtype=float)

        if not overwrite:
            x = x.copy()

        # Type casting!
        x = x.astype(float)

        # Step 1:
        for i in range(j, 0, -1):
            k = i - 1

            # Step 2:
            y[k] = (x[n + k] - np.dot(self.v(k), x[:n + k])) / self.δ(k)

            # Step 3:
            x[:n + k] -= b[k] * y[k]
            x[n + k]  -= d[k] * y[k]

        # Step 4: Solve core problem
        x[:n] = self.solver.solve_adj(x[:n])

        # Step 5:
        for k in range(0, j):
            # Step 6: test
            ypp       = (x[n + k] - np.dot(c[k], x[:n + k])) / self.ε(k)
            x[n + k]  = y[k]
            x[:n + k] -= self.w(k) * ypp
            x[n + k]  += ypp

        return x

    def solve(self, b):
        return self.solve_detail(self.m, b)

    def solve_adj(self, b):
        return self.solve_adj_detail(self.m, b)

    def solve_null(self):
        """ Solve where b = [f g] -> [0 1] """
        b = np.hstack((np.zeros(self.n + self.m - 1), 1))
        return self.solve_detail(self.m, b)

    def solve_null_adj(self):
        """ Solve where b = [f g] -> [0 1] """
        b = np.hstack((np.zeros(self.n + self.m - 1), 1))
        return self.solve_adj_detail(self.m, b)

    def solve_zero(self, f):
        """ Solve where b = [f g] -> [f 0] """
        return self.solve_detail(self.m, np.hstack((f, 0)))

    def solve_zero_adj(self, f):
        """ Solve where b = [f g] -> [f 0] """
        return self.solve_adj_detail(self.m, np.hstack((f, 0)))

    def min_norm(self, f):
        """
            Find the minimum norm solution of the underdetermined system

                [ A c ] (α β)^T = f      (1)

            We solve the system:

            / A c \  / x \  =  / f \
            |     |  |   |  =  |   |
            \ b d /  \ y /  =  \ 0 /

            and

            / A c \  / u \  =  / 0 \
            |     |  |   |  =  |   |
            \ b d /  \ v /  =  \ 1 /

        Then the general solution of the system (1) is given by:

            1. α = x + η u
            2. β = y + η v

        The minimum norm solution is in R[A^T] i.e. it must be orthogonal to N[A].
        Using this we derive the following condition:

                     x^T u + y v
            η = - ------------------
                    u^T u + v^T v

        """
        x   = self.solve_zero(f)
        t   = self.solve_null()
        eta = np.dot(x, t) / np.dot(t, t)
        return x - eta * t

    @property
    def is_singular(self):
        return self.solver.is_singular

    @property
    def shape(self):
        return (self.n + self.m, self.n + self.m, )

    @property
    def mat(self):
        n = self.n
        m = self.m
        new_shape = (n + m, n + m, )
        op = np.empty(new_shape, dtype=float)
        op[:n, :n] = self.solver.A
        for k in range(m):
            # Row
            op[n + k, :n + k] = self.bs[k]
            # Column
            op[:n + k, n + k] = self.cs[k]
            op[n + k, n + k]  = self.ds[k]

        return op
