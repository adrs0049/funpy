#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen

import numpy as np
import numbers
import scipy.linalg as LA
from scipy.sparse.linalg import spsolve_triangular

from .lucp import lucp


""" Computes the right nullvector of the matrix A, where N[A] = span{φ} """
def null_vector(A, right=True, left=False, normalize=True):
    # check that one of right or left is set
    assert right or left, 'One of left of right must be set!'

    # get the shape of A
    n, m = A.shape

    if np.any(np.iscomplex(A)):
        dtype = A.dtype
    elif issubclass(A.dtype.type, numbers.Integral):
        # make sure the matrix is float
        dtype = np.float64
        A = A.astype(dtype)
    else:
        dtype = A.dtype

    # compute decomposition such that LU = PAQ
    L, U, P, Q = lucp(A, full=False)

    if right:
        # the right eigenvector
        phi = np.negative(np.ones(m), dtype=dtype)

        # Solve for the right null-vector
        # Create the correct right hand side for the backwards solve
        phi[:m-1] = U.getcol(m-1).toarray().reshape(n)[:n-1]

        # then we must solve: Ustar = nu = u
        spsolve_triangular(U[:n-1, :m-1], phi[:m-1], lower=False, overwrite_b=True, overwrite_A=True)

        # now to compute phi we have to multiply by Q
        phi = phi[Q]  # Q.dot(phi)

        if normalize:
            phi /= np.linalg.norm(phi, ord=2)

    if left:
        # the left eigenvector
        psi = np.negative(np.ones(m), dtype=dtype)

        # Solve for the left null-vector
        # Create the correct right hand side for the backwards solve
        L_dual = L.getH()
        psi[:m-1] = L_dual.getcol(m-1).toarray().reshape(n)[:n-1]

        # then we must solve: Lstar w = l
        spsolve_triangular(L_dual[:n-1, :m-1], psi[:m-1], lower=False, overwrite_b=True, overwrite_A=True)

        # now to compute psi we have to multiply by P^-1
        psi = P.T.dot(psi)

        if normalize:
            psi /= np.linalg.norm(psi, ord=2)

    # can we solve this better?
    if left and right:
        return phi, psi
    elif left:
        return psi
    else:
        return phi

def right_null_vector(A, normalize=True):
    return null_vector(A, normalize=normalize, left=False, right=True)

def left_null_vector(A, normalize=True):
    return null_vector(A, normalize=normalize, left=True, right=False)

if __name__ == '__main__':
    A = np.diag([1,2,3])
    n = A.shape[0]
    es, ws = LA.eigh(A)

    print(50*'*')
    print('Test for matrix:\n%s.' % A)
    print(50*'*')
    print('Eigenvalues are:', es)
    print('Right-Eigenvectors are:\n', ws)
    print('Left-Eigenvectors are:\n', ws)

    for i, e in enumerate(es):
        # compute nullspace of the eigenvector problem
        W = A - e * np.eye(n)
        vec_r, vec_l = null_vector(W, left=True)

        print('Eigenvalue: %.2f associated to vec: %s.' % (e, vec_r))

        assert np.allclose(vec_r, ws[:, i]) or np.allclose(np.negative(vec_r), ws[:, i]), \
                'right_null_vector test failed!'

        assert np.allclose(vec_l, ws[:, i]) or np.allclose(np.negative(vec_l), ws[:, i]), \
                'left_null_vector test failed!'

    # second simple test
    A = np.arange(1,10,1).reshape((3,3))
    print(50*'*')
    print('Test for matrix:\n%s.' % A)
    print(50*'*')
    n = A.shape[0]
    es, wl, wr = LA.eig(A, left=True, right=True)
    es = np.real(es)

    print('Eigenvalues are:', es)
    print('Right-Eigenvectors are:\n', wr)
    print('Left-Eigenvectors are:\n', wl)

    for i, e in enumerate(es):
        # compute nullspace of the eigenvector problem
        W = A - e * np.eye(n)
        vec_r, vec_l = null_vector(W, left=True)

        print('Eigenvalue: %.3g associated to vec_r: %s.' % (e, vec_r))
        print('Eigenvalue: %.3g associated to vec_l: %s.' % (e, vec_l))

        B = np.vstack((vec_l, wl[:, i]))
        print('rank(Bl):', np.linalg.matrix_rank(B))

        B = np.vstack((vec_r, wr[:, i]))
        print('rank(Br):', np.linalg.matrix_rank(B))

        assert np.allclose(vec_r, wr[:, i]) or np.allclose(np.negative(vec_r), wr[:, i]), \
                'right_null_vector test failed!'

        assert np.allclose(vec_l, wl[:, i]) or np.allclose(np.negative(vec_l), wl[:, i]), \
                'left_null_vector test failed!'

    # Complex test
    A = np.array([[3, -2], [4, -1]])
    print(50*'*')
    print('Test for matrix:\n%s.' % A)
    print(50*'*')
    n = A.shape[0]
    es, wl, wr = LA.eig(A, left=True, right=True)

    print('Eigenvalues are:', es)
    print('Right-Eigenvectors are:\n', wr)
    print('Left-Eigenvectors are:\n', wl)

    for i, e in enumerate(es):
        # compute nullspace of the eigenvector problem
        W = A - e * np.eye(n)

        vec_r, vec_l = null_vector(W, left=True)

        print('Eigenvalue: %.3g associated to vec_r: %s.' % (e, vec_r))
        print('Eigenvalue: %.3g associated to vec_l: %s.' % (e, vec_l))

        B = np.vstack((vec_l, wl[:, i]))
        print('rank(Bl):', np.linalg.matrix_rank(B))

        B = np.vstack((vec_r, wr[:, i]))
        print('rank(Br):', np.linalg.matrix_rank(B))

        #print(f'Eigenvalue: {e:.3g} associated to vec: %s.' % vec)

        assert np.allclose(vec_r, wr[:, i]) or np.allclose(np.negative(vec_r), wr[:, i]), \
                'right_null_vector test failed!'

        assert np.allclose(vec_l, wl[:, i]) or np.allclose(np.negative(vec_l), wl[:, i]), \
                'left_null_vector test failed!'


    print('Simple tests passed.')
