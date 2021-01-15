#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
import scipy.sparse as sps

def csr_vappend(a,b):
    """ Takes in 2 csr_matrices and appends the second one to the bottom of the first one.
    Much faster than scipy.sparse.vstack but assumes the type to be csr and overwrites
    the first matrix instead of copying it. The data, indices, and indptr still get copied."""
    if not isinstance(a, sps.csr_matrix) or not isinstance(b, sps.csr_matrix):
        raise ValueError("Work only for CSR format -- use .tocsr() first!")
    a.data = np.hstack((a.data,b.data))
    a.indices = np.hstack((a.indices,b.indices))
    a.indptr = np.hstack((a.indptr,(b.indptr + a.nnz)[1:]))
    a._shape = (a.shape[0]+b.shape[0],b.shape[1])
    return a

def delete_row_csr(mat, i):
    if not isinstance(mat, sps.csr_matrix):
        raise ValueError("Work only for CSR format -- use .tocsr() first!")
    n = mat.indptr[i+1] - mat.indptr[i]
    if n > 0:
        mat.data[mat.indptr[i]:-n] = mat.data[mat.indptr[i+1]:]
        mat.data = mat.data[:-n]
        mat.indices[mat.indptr[i]:-n] = mat.indices[mat.indptr[i+1]:]
        mat.indices = mat.indices[i+1:]
    mat.indptr[i:-1] = mat.indptr[i+1:]
    mat.indptr[i:] -= n
    mat.indptr = mat.indptr[:-1]
    mat._shape = (mat._shape[0]-1, mat._shape[1])

def delete_rows_csr(mat, indices):
    """
        Deletes rows in indices from a csr sparse matrix
    """
    if not isinstance(mat, sps.csr_matrix):
        raise ValueError("Work only for CSR format -- use .tocsr() first!")
    indices = list(indices)
    mask = np.ones(mat.shape[0], dtype=bool)
    mask[indices] = False
    return mat[mask]

def eliminate_zeros_csr(mat, eps=None):
    """
        Eliminates matrix values that are below machine precision
    """
    if not isinstance(mat, sps.csr_matrix):
        raise ValueError("Work only for CSR format -- use .tocsr() first!")

    if eps is None:
        eps = np.finfo(mat.dtype).eps

    # simply set data elements to zero
    mat.data[np.abs(mat.data) < eps] = 0
    mat.eliminate_zeros()
    return mat

def flip_rows(mat, idenRows):
    """ Flips row in a sparse matrix: idenRows must be the order of the new rows """
    x = mat.tocoo()
    idenRows = np.argsort(idenRows)
    idenRows = np.asarray(idenRows, dtype=x.row.dtype)
    x.row = idenRows[x.row]
    return x.tocsr()
