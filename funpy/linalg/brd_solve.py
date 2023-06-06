#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np

import scipy.linalg as LA
import scipy.sparse.linalg as LAS
import scipy.sparse as sps

try:
    from scipy.sparse import csr_array
except ImportError:
    from scipy.sparse import csr_matrix as csr_array

from .qr import qr
from .qr_solve import QR
from .lucp import lucp



def permutation(arr, p):
    oarr = np.empty_like(arr)
    for i in range(arr.shape[0]):
        oarr[p[i]] = arr[i]

    return oarr


class BMatrix:
    def __init__(self, A, b, c, d, tol=1e-6):
        """
            / A c \  / x \ = / f \
            |      ||     |=|    |
            \ b d /  \ z / = \ h /

        """
        self.A = A
        self.b = b
        self.c = c
        self.d = d

        Q, R, C, k = qr(self.A, pivoting=True, mode='full')
        self.Q = Q
        self.R = R
        self.C = C
        self.P = np.argsort(C)

        # For debug only
        # self.p = csr_array((np.ones_like(self.P), (np.arange(0, n, 1), self.P)), shape=(n, n))
        # self.q = csr_array((np.ones_like(self.P), (self.P, np.arange(0, n, 1))), shape=(n, n))

        # Non-singular matrices
        self.Rbar = self.R[:-1, :-1]
        self.Sbar = self.R[:-1, -1]
        print('R = ', abs(self.R[-1, -1]))

        self.is_singular = abs(self.R[-1, -1]) < tol
        self.solve = self.solve_singular if self.is_singular else self.solve_nonsingular
        self.solve_null = self.solve_null_singular if self.is_singular else self.solve_null_nonsingular

        # Get eigenvectors
        self.phi = None
        self.psi = None

    @property
    def shape(self):
        return (self.A.shape[0] + 1, self.A.shape[1] + 1)

    @property
    def mat(self):
        n, m = self.A.shape
        op = np.empty(self.shape, dtype=float)
        op[:n, :m] = self.A
        op[m, :m]  = self.b
        op[:n, m]  = self.c
        op[n, m]   = self.d
        return op

    def _matvec(self, x):
        return np.hstack((self.A @ x[:-1] + self.c * x[-1],
                          np.dot(self.b, x[:-1]) + x[-1] * self.d))

    def _null(self):
        # Right null-vector
        y1  = LA.solve_triangular(self.Rbar, self.Sbar)
        phi = np.hstack((y1, -1))[self.P]

        # Left null-vector
        psi = self.Q[:, -1].conj().squeeze()
        return phi, psi

    def null(self):
        if self.phi is None or self.psi is None:
            self.phi, self.psi = self._null()

        return self.phi, self.psi

    def part(self, b):
        b = self.Q.T.dot(b)
        x = LA.solve_triangular(self.Rbar, b[:-1])
        return np.hstack((x, 0))[self.P]

    def solve_singular(self, b):
        # Get the null-vectors
        phi, psi = self.null()

        # Solve the system via the bordering technique
        f = b[:-1]
        h = b[-1]

        # Last component
        z = np.dot(psi, f) / np.dot(psi, self.c)

        # Get the particular solution
        xp = self.part(f - z * self.c)
        al = (h - self.d * z - np.dot(self.b, xp)) / np.dot(self.b, phi)
        x  = xp + al * phi

        return np.hstack((x, z))

    def solve_nonsingular(self, b):
        # Prepare
        beta  = LA.solve_triangular(self.R, self.b[self.C], trans='T')
        gamma = self.Q.T.dot(self.c)
        delta = self.d - np.dot(beta, gamma)

        # Solve the system
        f = b[:-1]
        h = b[-1]

        # Solve the inner system
        fhat = self.Q.T.dot(f)
        hhat = h - np.dot(beta, fhat)

        # Solve the outer system
        z = hhat / delta
        xhat = LA.solve_triangular(self.R, fhat - gamma * z, lower=False)
        return np.hstack((xhat[self.P], z))

    def solve_null_singular(self):
        # Get the null-vectors
        phi, psi = self.null()
        return np.hstack((phi, 0))

    def solve_null_nonsingular(self):
        # Prepare
        beta  = LA.solve_triangular(self.R, self.b[self.C], trans='T')
        gamma = self.Q.T.dot(self.c)
        delta = self.d - np.dot(beta, gamma)

        # Solve the system
        z = 1.0 / delta
        print('delta = ', delta)
        xhat = LA.solve_triangular(self.R, - z * gamma, lower=False)
        return np.hstack((xhat[self.P], z))


def solve(A, b, c, d, f):
    """
    Solve the system:

            / A  c \ / x \    / f \
            |      | |   |  = |   |
            \ b  d / \ z /    \ h /

        - A is a (n x n) matrix
        - c is a (n x 1) vector
        - b is a (n x 1) vector
        - d is a scalar

        - f is a (n+1 x 1) vector representing the right hand side

    """
    n = f.shape[0] - 1
    x = np.empty_like(f, dtype=float)

    # Compute the LU decomposition
    lu = LAS.splu(A)

    # Solve the using the bordering lemma
    gamma = LAS.spsolve_triangular(lu.L, permutation(c, lu.perm_r), lower=True)
    beta  = LAS.spsolve_triangular(lu.U.transpose(), permutation(b, lu.perm_c), lower=True)
    delta = d - np.dot(beta, gamma)
    fhat  = LAS.spsolve_triangular(lu.L, permutation(f[:n], lu.perm_r), lower=True)
    hhat  = f[n] - np.dot(beta, fhat)

    # Finally compute the solution
    x[n]  = hhat / delta
    x[:n] = LAS.spsolve_triangular(lu.U, fhat - x[n] * gamma, lower=False)[lu.perm_c]

    return x


def brd_solve_detail(lu, b, c, d, rhs, verify=True):
    """
        Solve the system:

        / A  c \ / x \   / f \
        |      | |   | = |   |
        \ b  d / \ z /   \ h /

        when the matrix A is rank 1 deficient. In this case
        z is expected to be zero if A is singular.
    """
    n, m = A.shape

    # make available
    L = lu.L
    U = lu.U
    perm_c = lu.perm_c
    perm_r = lu.perm_r

    # Step 2: Compute right null-vector
    phi       = np.empty(n, dtype=float)
    phi[:n-1] = U.getcol(m-1).toarray().reshape(n)[:n-1]
    phi[n-1]  = -1.0

    # Back-substitute
    phi[:m-1] = LAS.spsolve_triangular(U[:n-1, :m-1], phi[:n-1], lower=False, overwrite_b=True)

    # Apply permutation
    phi = phi[perm_c]

    if verify:
        print('Ax = ', np.max(np.abs(A @ phi)))

    # Step 3: Compute left null-vector
    psi       = np.empty(m, dtype=float)
    psi[:m-1] = L.getrow(m-1).toarray().reshape(m)[:m-1]
    psi[m-1]  = -1.0

    # Back substitute
    psi[:m-1] = LAS.spsolve_triangular(L.getH()[:n-1, :m-1], psi[:m-1], lower=False, overwrite_b=True)

    # Apply permutation
    psi = permutation(psi, perm_r)

    if verify:
        print('yA = ', np.max(np.abs(A.T @ psi)))

    # Step 4: Compute a particular solution of A
    xp     = np.empty(n, dtype=float)
    xp[:]  = rhs[:n]  # xp = f at the moment
    psi_f  = np.dot(psi, xp)
    psi_c  = np.dot(psi, c)
    z      = psi_f / psi_c
    xp[:] -= c * z

    # Back-substitute
    xp[:]    = LAS.spsolve_triangular(L, xp[perm_r], lower=True, overwrite_b=True)

    # Assemble the particular solution
    xp[:n-1] = LAS.spsolve_triangular(U[:n-1, :m-1], xp[:n-1], lower=False, overwrite_b=True)
    xp[n-1]  = 0.0

    # Apply permutation matrix
    xp = permutation(xp, perm_r)

    # Step 5: Compute the solution using the bordering technique
    alpha = (rhs[-1] - d * z - np.dot(b, xp)) / np.dot(b, phi)

    # Assemble solution
    rhs[:n] = xp + alpha * phi
    rhs[n]  = z

    # Return the solution (x z) and the left and right null-vectors of A
    return rhs, phi, psi


def brd_solve(lu, b, c, d, rhs, verify=True):
    """
        Solve the system:

        / A  c \ / x \   / f \
        |      | |   | = |   |
        \ b  d / \ z /   \ h /

        when the matrix A is rank 1 deficient. In this case
        z is expected to be zero if A is singular.
    """
    n, m = A.shape

    # make available
    L = lu.L
    U = lu.U
    perm_c = lu.perm_c
    perm_r = lu.perm_r

    # Step 2: Compute right null-vector
    phi       = np.empty(n, dtype=float)
    phi[:n-1] = U.getcol(m-1).toarray().reshape(n)[:n-1]
    phi[n-1]  = -1.0

    # Back-substitute
    phi[:m-1] = LAS.spsolve_triangular(U[:n-1, :m-1], phi[:n-1], lower=False, overwrite_b=True)

    # Apply permutation
    phi = phi[perm_c]

    if verify:
        print('Ax = ', np.max(np.abs(A @ phi)))

    # Step 3: Compute left null-vector
    psi       = np.empty(m, dtype=float)
    psi[:m-1] = L.getrow(m-1).toarray().reshape(m)[:m-1]
    psi[m-1]  = -1.0

    # Back substitute
    psi[:m-1] = LAS.spsolve_triangular(L.getH()[:n-1, :m-1], psi[:m-1], lower=False, overwrite_b=True)

    # Apply permutation
    psi = permutation(psi, perm_r)

    if verify:
        print('yA = ', np.max(np.abs(A.T @ psi)))

    # Step 4: Compute a particular solution of A
    xp     = np.empty(n, dtype=float)
    xp[:]  = rhs[:n]  # xp = f at the moment
    psi_f  = np.dot(psi, xp)
    psi_c  = np.dot(psi, c)
    z      = psi_f / psi_c
    xp[:] -= c * z

    # Back-substitute
    xp[:]    = LAS.spsolve_triangular(L, xp[perm_r], lower=True, overwrite_b=True)

    # Assemble the particular solution
    xp[:n-1] = LAS.spsolve_triangular(U[:n-1, :m-1], xp[:n-1], lower=False, overwrite_b=True)
    xp[n-1]  = 0.0

    # Apply permutation matrix
    xp = permutation(xp, perm_r)

    # Step 5: Compute the solution using the bordering technique
    alpha = (rhs[-1] - d * z - np.dot(b, xp)) / np.dot(b, phi)

    # Assemble solution
    rhs[:n] = xp + alpha * phi
    rhs[n]  = z

    # Return the solution (x z) and the left and right null-vectors of A
    return rhs, phi, psi


def solve_nonsing(lu, b, c, d):
    """
    Solve the system:

            / A    c \ / v \    / 0 \
            |        | |   |  = |   |
            \ b^T  d / \ h /    \ 1 /

        - A is a (n x n) matrix
        - c is a (n x 1) vector
        - b is a (n x 1) vector
        - d is a scalar

    System 1:

        1. L[γ] = Pc
        2. U^T[β] = Q^T b
        3. δ = d - β^T γ
        4. z = 1 / δ
        5. U χ = - γ / δ
        6. x = Q χ

    """
    n, m = lu.shape

    # output vectors
    v = np.empty(1 + n, dtype=float)

    # Solve system using the bordering technique
    gamma = LAS.spsolve_triangular(lu.L, permutation(c, lu.perm_r), lower=True)
    beta  = LAS.spsolve_triangular(lu.U.transpose(), permutation(b, lu.perm_c), lower=True)
    delta = d - np.dot(beta, gamma)

    # Solve the decomposed system
    v[n]  = 1.0 / delta
    v[:n] = LAS.spsolve_triangular(lu.U, -v[n] * gamma, lower=False)[lu.perm_c]

    return v


def solve_sing(lu, b, c, d, A=None):
    """
        Solve the system:

        / A    c \ / v \   / 0 \
        |        | |   | = |   |
        \ b^T  d / \ h /   \ 1 /

        when the matrix A is rank 1 deficient. In this case
        h is expected to be zero if A is singular.
    """
    n, m = lu.shape

    # output vectors
    v = np.empty(1 + n, dtype=float)

    # make available
    L = lu.L
    U = lu.U
    perm_c = lu.perm_c
    perm_r = lu.perm_r

    # Step 2: Compute right null-vector of A
    phi       = np.empty(n, dtype=float)
    phi[:n-1] = U.getcol(m-1).toarray().reshape(n)[:n-1]
    phi[n-1]  = -1.0

    # Back-substitute
    phi[:m-1] = LAS.spsolve_triangular(U[:n-1, :m-1], phi[:n-1], lower=False, overwrite_b=True)

    # Apply permutation
    phi = phi[perm_c]

    # Step 3: Compute left null-vector of A
    psi       = np.empty(m, dtype=float)
    psi[:m-1] = L.getrow(m-1).toarray().reshape(m)[:m-1]
    psi[m-1]  = -1.0

    # Back substitute
    psi[:m-1] = LAS.spsolve_triangular(L.getH()[:n-1, :m-1], psi[:m-1], lower=False, overwrite_b=True)

    # Apply permutation
    psi = permutation(psi, perm_r)

    # Step 4: Compute a particular solution of system 1
    #
    #           ψ^T f
    #   1) z = -------
    #           ψ^T c
    #
    #   2) Solve Ax = f - z c
    #       -> Solution x = x_p + α φ
    #
    #   3) Find particular solution of A
    #
    #          h - d z - b^T x_p
    #   4) α = -----------------
    #               b^T φ
    #
    xp      = np.zeros(n, dtype=float)
    psi_f   = np.dot(psi, xp)
    psi_c   = np.dot(psi, c)
    z       = psi_f / psi_c
    xp[:]  -= c * z

    # Back-substitute
    xp[:]    = LAS.spsolve_triangular(L, xp[perm_r], lower=True, overwrite_b=True)

    # Assemble the particular solution
    xp[:n-1] = LAS.spsolve_triangular(U[:n-1, :m-1], xp[:n-1], lower=False, overwrite_b=True)
    xp[n-1]  = 0.0

    # Apply permutation matrix
    xp = permutation(xp, perm_r)

    # Step 5: Compute the solution using the bordering technique
    alpha = (1.0 - d * z - np.dot(b, xp)) / np.dot(b, phi)

    # Assemble solution
    v[:n] = xp + alpha * phi
    v[n]  = z

    # Return the solution (x z) and the left and right null-vectors of A
    return v


class LUCP:
    def __init__(self, A):
        l, u, p, q = lucp(A)
        self.L = l
        self.U = u
        self.perm_c = q
        self.perm_r = p

    @property
    def shape(self):
        return self.L.shape


def nullvector(A):
    n, m = A.shape

    # Step 1: Compute LU decomposition
    lu = LUCP(A.toarray())

    # make available
    U = lu.U
    perm_c = lu.perm_c

    # Step 2: Compute right null-vector of A
    phi       = np.empty(n, dtype=float)
    phi[:n-1] = U.getcol(m-1).toarray().reshape(n)[:n-1]
    phi[n-1]  = -1.0

    # Back-substitute
    phi[:m-1] = LAS.spsolve_triangular(U[:n-1, :m-1], phi[:n-1], lower=False, overwrite_b=True)

    # Apply permutation
    phi = phi[perm_c]

    return phi


def brd_null_simp(A, b, c, d, eps=1e-8, verify=True):
    """
        Solve the system:

        / A  c \ / x \   / 0 \
        |      | |   | = |   |
        \ b  d / \ z /   \ 1 /

        There are two cases

            1) The matrix A is nonsingular
            2) The matrix A is rank 1 deficient. In this case z is expected to be zero.

    """
    n, m = A.shape

    # Step 1: Compute LU decomposition
    # lu = LAS.splu(A)
    lu = LUCP(A.toarray())
    is_singular = abs(lu.U[n - 1, n - 1]) < eps
    return solve_sing(lu, b, c, d, A=A) if is_singular else solve_nonsing(lu, b, c, d)


def solve_null_nonsing(lu, b, c, d):
    """
    Solve the system:

            / A    c \ / v \    / 0 \
            |        | |   |  = |   |
            \ b^T  d / \ h /    \ 1 /

    and the system:

            / A^T  b \ / w \    / 0 \
            |        | |   |  = |   |
            \ c^T  d / \ h /    \ 1 /

        - A is a (n x n) matrix
        - c is a (n x 1) vector
        - b is a (n x 1) vector
        - d is a scalar

    System 1:                   System 2:

        1. L[γ] = Pc                U^T[γ] = Q^T b
        2. U^T[β] = Q^T b           L[β] = Pc
        3. δ = d - β^T γ            δ = d - β^T γ
        4. z = 1 / δ                z = 1 / δ
        5. U χ = - γ / δ            L^T[χ] = - γ / δ
        6. x = Q χ                  x = P^T χ

    """
    n, m = lu.shape

    # output vectors
    v = np.empty(1 + n, dtype=float)
    w = np.empty(1 + n, dtype=float)

    # Solve system 1 using the bordering technique
    gamma = LAS.spsolve_triangular(lu.L, permutation(c, lu.perm_r), lower=True)
    beta  = LAS.spsolve_triangular(lu.U.transpose(), permutation(b, lu.perm_c), lower=True)
    delta = d - np.dot(beta, gamma)

    # Solve the decomposed system
    v[n]  = 1.0 / delta
    v[:n] = LAS.spsolve_triangular(lu.U, -v[n] * gamma, lower=False)[lu.perm_c]

    # Solve system 2 using the bordering technique
    gamma = LAS.spsolve_triangular(lu.U.transpose(), permutation(b, lu.perm_c), lower=True)
    beta  = LAS.spsolve_triangular(lu.L, permutation(c, lu.perm_r), lower=True)
    delta = d - np.dot(beta, gamma)

    # Solve the decomposed system
    w[n]  = 1.0 / delta
    w[:n] = LAS.spsolve_triangular(lu.L.transpose(), -w[n] * gamma, lower=False)[lu.perm_r]

    return v, w


def brd_null(A, b, c, d, eps=1e-4, verify=True):
    """
        Solve the system:

        / A  c \ / x \   / 0 \
        |      | |   | = |   |
        \ b  d / \ z /   \ 1 /

        There are two cases

            1) The matrix A is nonsingular
            2) The matrix A is rank 1 deficient. In this case z is expected to be zero.

    """
    n, m = A.shape

    # Step 1: Compute LU decomposition
    lu = LAS.splu(A)

    is_singular = lu.U[n-1, n-1] < eps
    return solve_null_sing(lu, b, c, d) if is_singular else solve_null_nonsing(lu, b, c, d)


def test_simple():
    A = np.array([[1, 2, 3], [1, 0, 1], [3, 2, 0]])
    d = 10.0
    b = 0 + np.arange(3)
    c = 6 + np.arange(3)

    print('b = ', b)
    print('c = ', c)

    # Total matrix
    B = np.empty((4, 4), dtype=float)
    B[:3, :3] = A
    B[3, :3] = b
    B[:3, 3] = c
    B[3, 3] = d

    print('B = ')
    print(B)

    # rhs
    f = np.arange(4)

    print('A = ')
    print(A)

    x = solve(A, b, c, d, f)
    print('x = ', x)
    print('res = ', np.max(np.abs(B @ x - f)))

    x = LA.solve(B, f)
    print('x = ', x)
    print('res = ', np.max(np.abs(B @ x - f)))


def test_simple2():
    A = np.array([[1, 2, 3], [1, 0, 1], [3, 2, 0]])
    d = 10.0
    b = 0 + np.arange(3)
    c = 6 + np.arange(3)

    print('b = ', b)
    print('c = ', c)

    # Total matrix
    B = np.empty((4, 4), dtype=float)
    B[:3, :3] = A
    B[3, :3] = b
    B[:3, 3] = c
    B[3, 3] = d

    print('B = ')
    print(B)

    # rhs
    f = np.zeros(4)
    f[-1] = 1.0

    print('A = ')
    print(A)

    x = solve_null(A, b, c, d)
    print('x = ', x)
    print('res = ', np.max(np.abs(B @ x - f)))

    x = LA.solve(B, f)
    print('x = ', x)
    print('res = ', np.max(np.abs(B @ x - f)))


# A simple test
if __name__ == '__main__':
    print('Simple Test!')
    test_simple()

    print('Simple Test 2!')
    test_simple2()
