#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
import scipy
import operator
import numbers
import h5py as h5
from copy import deepcopy
from numbers import Number
from numpy.core.multiarray import normalize_axis_index
from scipy.fftpack import ifft, fft, dct, idct, idst, ifftshift

# Local imports
from support.cached_property import lazy_property
from cheb.chebpts import chebpts_type2, chebpts_type2_compute, barymat, quadwts
from cheb.refine import RefineBase, Refine, RefineCompose1, RefineCompose2, FunctionContainer
from cheb.detail import polyfit, polyval, clenshaw, roots
from cheb.detail import standardChop, prolong, simplify_coeffs, happiness_check
from cheb.diff import computeDerCoeffs
import cheb.qr
from cheb.minmax import minmaxCol

EPS = np.finfo(float).eps


class chebtec(np.lib.mixins.NDArrayOperatorsMixin):
    """ This needs some information still! Currently only type-2 polynomials """
    def __init__(self, op=None, values=None, eps=EPS, coeffs=None, *args, **kwargs):
        """ Improve this so that we can just pass in one object and we figure out what it is """
        self.coeffs = coeffs

        # tolerance
        self.eps = eps

        # add varying interval support!
        self.hscale = kwargs.pop('hscale', 1)
        self.maxLength = kwargs.pop('maxLength', 2**14)
        self.ishappy = kwargs.pop('ishappy', False)

        if values is None and self.coeffs is None and op is None:
            self.coeffs = np.zeros((0, 0), order='F')
        elif op is not None:
            self.coeffs = np.zeros((0,0), order='F')

            # Treat numpy arrays differently from things we can call!
            if isinstance(op, np.ndarray):
                if len(op.shape)==1:
                    op = np.expand_dims(op, axis=1)
                self.coeffs = polyfit(op)
            else:
                # Create callable container
                if isinstance(op, list):
                    op = FunctionContainer(op)
                # check what kind of op we have
                if isinstance(op, RefineBase):
                    self.populate(op)
                else:  # Assume that it's a lambda -> TODO check that
                    refine = Refine(op=op, minSamples=min(self.maxLength, kwargs.pop('minSamples', 17)),
                                    strategy=kwargs.pop('resample', 'nested'))
                    self.populate(refine)
        elif self.coeffs is None:
            self.coeffs = polyfit(values)

        # Update the happiness status
        if not self.ishappy:
            self.ishappy = self.happy()

        # call simplify - this will also check whether I am happy
        simplify = kwargs.pop('simplify', True)
        if simplify and self.size > 0:
            self.simplify()

    @classmethod
    def from_hdf5(cls, hdf5_file):
        coeffs = np.asfortranarray(hdf5_file[:])
        eps = hdf5_file.attrs["eps"]
        hscale = hdf5_file.attrs["hscale"]
        maxLength = hdf5_file.attrs["maxLength"]
        ishappy = hdf5_file.attrs["ishappy"]
        return cls(coeffs=coeffs, eps=eps, hscale=hscale, maxLength=maxLength,
                   ishappy=ishappy, simplify=False)

    @classmethod
    def from_values(cls, values, *args, **kwargs):
        coeffs = polyfit(values)
        return cls(coeffs=coeffs, *args, **kwargs)

    def __deepcopy__(self, memo):
        id_self = id(self)
        _copy = memo.get(id_self)
        if _copy is None:
            _copy = type(self)(
                coeffs=deepcopy(self.coeffs),
                eps=self.eps, hscale=self.eps,
                maxLength=self.maxLength,
                simplify=False,
                ishappy=self.ishappy)
            memo[id_self] = _copy
        return _copy

    def __repr__(self):
        return str(self.coeffs.T)

    def __str__(self):
        return str(self.coeffs.T)

    def __len__(self):
        return self.coeffs.shape[0]

    def __array__(self):
        return self.coeffs.flatten(order='F')

    def __array_ufunc__(self, numpy_ufunc, method, *inputs, **kwargs):
        import cheb.ufuncs as cp_funcs

        # out = kwargs.get('out', ())
        # TODO: this probably needs to be somewhere else it can't check for trigtech input!
        #for x in inputs + out:
        #    if not isinstance(x, (np.ndarray, Number, type(self))):
        #        return NotImplemented

        if method == "__call__":
            name = numpy_ufunc.__name__

            try:
                cp_func = getattr(cp_funcs, name)
            except AttributeError:
                pass
            else:
                return cp_func(*inputs, **kwargs)

            # If we don't have a special implementation we default to evaluating by value!
            if len(inputs) == 1:
                return chebtec(op=lambda x: numpy_ufunc(inputs[0](x)))
            elif len(inputs) == 2:
                return chebtec(op=lambda x: numpy_ufunc(inputs[0](x), inputs[1](x)))
            else:
                return NotImplemented
        else:
            return NotImplemented

    def isfortran(self):
        return self.coeffs.flags.f_contiguous

    @lazy_property
    def vscale(self):
        """ Estimates the vertical scale of a function """
        if self.coeffs is None:
            return 0
        elif self.coeffs.shape[0] == 1:
            return np.abs(self.coeffs[0, :])
        else:
            return np.max(np.abs(polyval(self.coeffs)), axis=0)

    @property
    def ValsDisc(self):
        return "chebcolloc2"

    @property
    def nbytes(self):
        return self.coeffs.nbytes

    @property
    def istrig(self):
        return False

    @property
    def ndim(self):
        return self.coeffs.ndim

    @property
    def size(self):
        return self.coeffs.shape[1]

    """ This is not elegant yet """
    def __matmul__(self, other):
        return self.coeffs @ other

    @property
    def T(self):
        return self.coeffs.T

    @property
    def shape(self):
        return self.coeffs.shape

    @property
    def type(self):
        return 'cheb'

    """ This is the number of values or coefficients """
    @property
    def n(self):
        return self.shape[0]

    """ Useful to select a column of function """
    def __getitem__(self, idx):
        #assert idx >= 0 and idx < self.m, 'Index %d out of range [0, %d].' % (idx, self.m-1)
        if idx < 0 or idx >= self.m:
            raise IndexError
        return chebtec(coeffs=self.coeffs[:, None, idx], simplify=False, ishappy=self.ishappy)

    """ This is the number of chebyshev polynomials stored in this class """
    @property
    def m(self):
        if len(self.shape) == 1:
            return 1
        return self.shape[1]

    """ Return the points at which the chebtec is sampled at """
    @property
    def x(self):
        return chebpts_type2_compute(self.n)

    @property
    def values(self):
        return polyval(self.coeffs)

    def happy(self):
        tol = np.maximum(self.hscale, self.vscale) * self.eps * np.ones(self.m)
        ishappy, cutoff = happiness_check(self.coeffs, tol)
        return bool(ishappy), cutoff

    def chebpts(self, n=None):
        if n is None: n = self.n
        return chebpts_type2_compute(self.n)

    def get_values(self):
        return polyval(self.coeffs)

    """ Construct a chebtech from a callable op """
    def populate(self, refine):
        while True:
            values, giveUp = refine(self)

            # get the coefficients
            self.coeffs = polyfit(values)

            if giveUp:
                break

            # check happiness
            ishappy, cutoff = self.happy()

            if ishappy:
                self.prolong(cutoff+1)
                break

    """ Evaluates chebtec at x = -1 """
    def lval(self):
        c = np.copy(self.coeffs)
        c[1::2] *= -1
        return np.sum(c, axis=0)

    """ Evaluate chebtec at x = 1 """
    def rval(self):
        return np.sum(self.coeffs, axis=0)

    """ TODO: add support to get other kind of coefficients """
    def chebcoeff(self):
        return self.coeffs

    def restrict(self, s):
        """ Restrict the chebtec to a subinterval s """
        s = np.asanyarray(s)

        # check that we really have a subinterval
        if s[0] < -1 or s[1] > 1 or np.any(np.diff(s) <= 0):
            assert False, 'Not a valid interval'
        elif s.size == 2 and np.all(s == np.asarray([-1, 1])):
            # nothing to do here
            return self

        n, m = self.shape
        numInts = s.size - 1

        # compute values on new grid
        y = 0.5 * np.vstack((1-self.x, 1+self.x)).T @ np.vstack((s[:-1], s[1:]))  # new grid
        values = self.feval(y.flatten())  # values on new grid

        # must rearrange the order of columns
        if m > 1:
            numCols = m * numInts
            index = np.reshape(np.reshape(np.arange(numCols), (numInts, m)).T, numCols)
            values = values[:, index]

        # generate new coefficients
        coeffs = polyfit(values)
        return chebtec(values=values, coeffs=coeffs)


    def prolong(self, Nout):
        # If Nout < length(self) -> compressed by chopping
        # If Nout > length(self) -> coefficients are padded by zero
        self.coeffs = np.asarray(prolong(self.coeffs, Nout), order='F')
        return self

    def prolong_coeffs(self, Nout):
        # TODO fix the endless duplication of this code-path!
        """ Return the prolonged coefficients only """
        return np.asarray(prolong(self.coeffs, Nout), order='F')

    def simplify(self, eps=None, force=False):
        # if not happy simply do nothing
        if not force and not self.ishappy:
            return self

        self.coeffs = simplify_coeffs(self.coeffs, eps=self.eps if eps is None else eps)
        return self

    def __call__(self, other):
        """
        IF other is a ndarray: -> Evaluate the chebtech at those points.
        IF other is a chebtech: -> Compose the chebtech
        """
        return self.feval(other)

    def feval(self, x):
        #n = len(self)
        #if n <= 4000 or x.size <= 4000:
        return clenshaw(x, self.coeffs).squeeze()
        #else: DCT

    def points(self):
        points = chebpts_type2
        return points(self.n)

    def eval(self, y):
        """ Construct the evaluation operator to compute the values of the
        chebtech at the points y.
        """
        x, _, v, _ = self.points()
        # construct
        return barymat(y, x, v)

    def flipud(self):
        """ Flip / reverse a chebtec object such that G(x) = F(-x) for all x in [-1, 1] """
        coeffs = np.copy(self.coeffs)
        coeffs[1::2] *= -1
        return chebtec(coeffs=coeffs)

    def fliplr(self):
        """ Flip columns of an array-valued chebtec object. """
        return chebtec(coeffs=np.fliplr(self.coeffs))

    def diff(self, k=1, axis=0):
        """ Compute the k-th derivative of the chebtec f """
        n = self.coeffs.shape[0]

        # return zero if differentiating too much
        if k >= n:
            return chebtec(coeffs=np.zeros_like(self.coeffs))

        # Iteratively compute the coefficients of the derivatives
        c = computeDerCoeffs(self.coeffs)
        for _ in range(1, k):
            c = computeDerCoeffs(c)

        # This returns a chebtec that has only N - k Chebyshev coefficients -> note that setting
        # happy will cause the function to be simplified!
        return chebtec(coeffs=c, ishappy=self.ishappy, simplify=False)

    """ Definite Integral """
    def sum(self, axis=None, **unused_kwargs):
        """ Definite integral of a chebtec f on the interval [-1, 1].

        If f is an array-valued chebtec, then the result is a row vector
        containing the definite integrals of each column.

        """
        n = self.n

        # Constant chebtec
        if n == 1:
            return 2 * self.coeffs

        # Evaluate the integral by using the Chebyshev coefficients
        #
        # Int_{-1}^{1} T_k(x) dx = 2 / (1 - k^2)   if k even
        # Int_{-1}^{1} T_k(x) dx = 0               if k odd
        #
        # Thm 19.2 in Trefethen
        mask = np.ones_like(self.coeffs)
        mask[1::2, :] = 0
        out = np.expand_dims(np.hstack((2, 0, 2 / (1 - np.arange(2, n)**2))), axis=0) @ (self.coeffs * mask)
        return out.squeeze()

    """ Indefinite Integral """
    def cumsum(self, m=1, **unused_kwargs):
        """ Indefinite integral of chebtec f, with the constant of integration
        chosen such that f(-1) = 0.

        Given a Chebyshev polynomial of length n, we have that

            f(x) = sum_{j = 0}^{n-1} c_j T_j(x)

        Its integral is represented by a polynomial of length n+1 given by

            g(x) = sum_{j = 0}^{n} b_j T_j(x)

        with b_0 = sum_{j = 1}^{n} (-1)^{j+1} b_j

        the other coefficients are:
            b_1 = c_0 - c_2 / 2
            b_r = (c_{r-1} - c_{r+1})/(2r) for r >

        with c_{n+1} = c_{n+2} = 0
        """
        n, m = self.shape
        c = np.vstack((self.coeffs, np.zeros((2, m))))  # pad with zeros
        b = np.zeros((n+1, m))

        # compute b_(2) ... b_(n+1)
        b[2:n+1, :] = (c[1:n, :] - c[3:n+2, :]) / np.tile(np.expand_dims(2*np.arange(2, n+1), axis=1), (1, m))
        # compute b(1)
        b[1, :] = c[0, :] - c[2, :] / 2
        v = np.ones((1, n))
        v[:, 1::2] = -1
        b[0, :] = np.matmul(v, b[1:, :])

        # Create the new chebtec
        g = chebtec(coeffs=b)

        # simplify
        g = g.simplify()

        # ensure that f(-1) = 0
        lval = g.lval()
        g.coeffs[0, :] = g.coeffs[0, :] - lval
        return g

    """ Get the conjugate """
    def conj(self):
        nc = np.conj(self.coeffs)
        return chebtec(coeffs=nc, simplify=False)

    def __eq__(self, other):
        return np.all(self.shape == other.shape) and np.all(self.coeffs == other.coeffs)

    def vscl(self):
        """ Estimate the vertical scale of a function. """
        c = self.coeffs
        if c.size == 0:
            return 0
        elif c.shape[0] == 1:
            return np.abs(c).squeeze()
        else:
            vals = polyval(c)
            return np.max(np.max(np.abs(vals), axis=0))

    """ This cannot be a member function """
    def innerproduct(self, other):
        """ Computes the L2 inner product on [-1, 1] of two Chebyshev series """
        n = len(self) + len(other)

        fvalues = polyval(prolong(self.coeffs, n))
        gvalues = polyval(prolong(other.coeffs, n))

        # compute Clenshaw-Curtis quadrature weights
        w = quadwts(n)

        # compute the inner-product
        out = np.matmul(fvalues.T * w, gvalues)

        # force non-negative output if the inputs are equal
        # if isequal(f, g):
        # dout = diag(diag(out))
        # out = out - dout + abs(dout)

        return out.squeeze()

    def compose(self, op, g=None):
        """ Returns a lambda generating the composition of the two functions """
        if g is None:
            """ Compute op(f) """
            return lambda x: op(self(x))
        else:
            """ Compute op(f, g) """
            return lambda x: op(self(x), g(x))

    def roots(self, *args, **kwargs):
        # If we don't simplify this may lead to crashes!
        self.simplify()
        return roots(self)

    def minandmax(self, *args, **kwargs):
        fp = self.diff()
        x = self.x

        if self.m == 1:
            vals, pos = minmaxCol(self, fp, x)
        else:
            vals = np.zeros((2, self.m))
            pos = np.zeros((2, self.m))
            for i in range(self.m):
                vals[:, i, None], pos[:, i, None] = minmaxCol(self[i], fp[i], x)

        return vals, pos

    def qr(self, *args, **kwargs):
        Q, R = cheb.qr.qr(self, *args, **kwargs)
        Q = chebtec(coeffs=Q, simplify=False)
        return Q, R

    """ I/O support """
    def writeHDF5(self, fh):
        # Write the coefficients
        fd = fh.create_dataset('poly', data=self.coeffs)
        fd.attrs["eps"] = self.eps
        fd.attrs["hscale"] = self.hscale
        fd.attrs["maxLength"] = self.maxLength
        fd.attrs["ishappy"] = self.ishappy

    def readHDF5(self, fh):
        self.coeffs = np.asfortranarray(fh[:]) # TODO FIXME
        self.eps = fh.attrs["eps"]
        self.hscale = fh.attrs["hscale"]
        self.maxLength = fh.attrs["maxLength"]
        self.ishappy = fh.attrs["ishappy"]


def compose(f, op, g=None):
    if g is None:
        resampler = RefineCompose1(f, op)
        return chebtec(op=resampler)
    else:
        resampler = RefineCompose2(f, op, g)
        return chebtec(op=resampler)
