#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np

def abstractQR(A, E, inner_product, norm, tol):
    numCols = A.shape[1]
    R = np.zeros((numCols, numCols))
    V = np.copy(A)

    for k in range(numCols):
        I = np.arange(0, k-1)
        J = np.arange(k+1, numCols)

        # scale
        scl = max(norm(E[:, k]), norm(A[:, k]))

        # Multiply the k-th column of A with the basis in E:
        ex = inner_product(E[:, k], A[:, k])
        aex = abs(ex)

        # Adjust the sign of the k-th column in E
        if aex < tol * scl:
            s = 1
        else:
            s = -np.sign(ex / aex)
        E[:, k] *= s

        # Compute the norm of the k-th column of A:
        r = np.sqrt(inner_product(A[:, k], A[:, k]))
        R[k, k] = r

        # compute the reflection of v
        v = r * E[:, k] - A[:, k]

        # Make it more orthogonal
        for i in I:
            ev = inner_product(E[:, i], v)
            v -= E[:, i] * ev

        # normalize
        nv = np.sqrt(inner_product(v, v))
        if nv < tol * scl:
            v = E[:, k]
        else:
            v /= nv

        # Store
        V[:, k] = v

        # Subtract v from the remaining columns of A:
        for j in J:
            # Apply Householder reflection
            av = inner_product(v, A[:, j])
            A[:, j] = A[:, j] - 2 * v * av

            # Compute other non-zero entries in the current row and store them
            rr = inner_product(E[:, k], A[:, j])
            R[k, j] = rr

            # subtract off projections onto the current value
            A[:, j] = A[:, j] - E[:, k] * rr

    # Form Q from the columns of V:
    Q = E
    for k in range(numCols-1, -1, -1):
        for j in range(k, numCols):
            # Apply reflection again
            vq = inner_product(V[:, k], Q[:, j])
            Q[:, j] -= 2 * (V[:, k] * vq)

    return Q, R
