#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
from copy import deepcopy
from numbers import Number

import numpy as np
from numpy.core.multiarray import normalize_axis_index

try:
    # Import compiled components -> Make sure this happens before all imports below!
    from cheb.refine import RefineBase, Refine, RefineCompose1, RefineCompose2, FunctionContainer
    from cheb.detail import polyfit, polyval, clenshaw, roots
    from cheb.detail import standardChop, prolong, simplify_coeffs, happiness_check

except ImportError:
    from .build_cheb import build_cheb_module
    build_cheb_module()

    # Try the imports again
    try:
        from .refine import RefineBase, Refine, RefineCompose1, RefineCompose2, FunctionContainer
        from .detail import polyfit, polyval
        from .detail import clenshaw, roots
        from .detail import standardChop, prolong, simplify_coeffs, happiness_check
    except Exception as e:
        raise e


# Local imports
from ..support.cached_property import lazy_property
from .pts import chebpts_type2, chebpts_type2_compute, barymat, quadwts
from .diff import computeDerCoeffs
from .qr import qr
from .minmax import minmaxCol

# Directory for numpy implementation of functions
HANDLED_FUNCTIONS = {}


class chebtech(np.lib.mixins.NDArrayOperatorsMixin):
    """ This needs some information still! Currently only type-2 polynomials """
    def __init__(self, op=None, *args, **kwargs):
        """ Improve this so that we can just pass in one object and we figure out what it is """
        self.coeffs = kwargs.pop('coeffs', np.zeros((0, 0), order='F'))

        # tolerance
        self.eps = kwargs.pop('eps', np.finfo(float).eps)

        # add varying interval support!
        self.hscale = kwargs.pop('hscale', 1)
        self.maxLength = kwargs.pop('maxLength', 1 + 2**14)
        self.ishappy = kwargs.pop('ishappy', False)

        if op is not None:
            self.coeffs = np.zeros((0, 0), order='F')

            # Treat numpy arrays differently from things we can call!
            if isinstance(op, np.ndarray):
                if len(op.shape) == 1:
                    op = np.expand_dims(op, axis=1)
                self.coeffs = polyfit(op)
            else:
                # Create callable container
                if isinstance(op, list):
                    op = FunctionContainer(op) if len(op) > 1 else op[0]
                # check what kind of op we have
                if isinstance(op, RefineBase):
                    self.populate(op)
                else:  # Assume that it's a lambda -> TODO check that
                    refine = Refine(op=op, minSamples=min(self.maxLength, kwargs.pop('minSamples', 17)),
                                    strategy=kwargs.pop('resample', 'nested'))
                    self.populate(refine)

        # Update the happiness status
        assert self.coeffs.size > 0, 'Something went wrong during chebfun construction!'
        if not self.ishappy:
            self.ishappy, _ = self.happy()

        # call simplify - this will also check whether I am happy
        simplify = kwargs.pop('simplify', True)
        if simplify and self.size > 0:
            self.simplify()

    @classmethod
    def from_hdf5(cls, hdf5_file):
        coeffs = np.asfortranarray(hdf5_file[:])  # TODO FIXME
        eps = hdf5_file.attrs["eps"]
        hscale = hdf5_file.attrs["hscale"]
        maxLength = hdf5_file.attrs["maxLength"]
        ishappy = hdf5_file.attrs["ishappy"]
        return cls(coeffs=coeffs, eps=eps, hscale=hscale, maxLength=maxLength,
                   ishappy=ishappy, simplify=False)

    @classmethod
    def from_values(cls, values, *args, **kwargs):
        coeffs = polyfit(values)
        return cls(coeffs=coeffs, ishappy=True, *args, **kwargs)

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
        if self.m == 1:
            return np.expand_dims(self.coeffs.flatten(order='F'), axis=1)
        else:
            return self.coeffs.flatten(order='F')

    def __array_ufunc__(self, numpy_ufunc, method, *inputs, **kwargs):
        from . import ufuncs as cp_funcs

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
                return chebtech(op=lambda x: numpy_ufunc(inputs[0](x)),
                               eps=self.eps, maxLength=self.maxLength)
            elif len(inputs) == 2:
                return chebtech(op=lambda x: numpy_ufunc(inputs[0](x), inputs[1](x)),
                               eps=self.eps, maxLength=self.maxLength)
            else:
                return NotImplemented
        else:
            return NotImplemented

    """ Implement array ufunc support """
    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def isfortran(self):
        return self.coeffs.flags.f_contiguous

    @lazy_property
    def vscale(self):
        """ Estimates the vertical scale of a function """
        if self.coeffs is None or self.coeffs.size == 0:
            return 0
        elif self.coeffs.shape[0] == 1:
            return np.abs(self.coeffs[0, :])
        else:
            return np.max(np.abs(polyval(self.coeffs)), axis=0)

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
        if isinstance(idx, slice):
            if idx.start < 0 or idx.stop > self.m:
                raise IndexError("The index slice({0:d}, {1:d}) is out of range ({2:d})!".format(idx.start, idx.stop, self.m))

            return chebtech(coeffs=self.coeffs[:, idx], eps=self.eps,
                           maxLength=self.maxLength, simplify=False,
                           ishappy=self.ishappy)

        elif isinstance(idx, int):
            if idx < 0 or idx >= self.m:
                raise IndexError("The index {0:d} is out of range ({1:d})!".format(idx, self.m))

            return chebtech(coeffs=self.coeffs[:, None, idx], eps=self.eps,
                           maxLength=self.maxLength, simplify=False,
                           ishappy=self.ishappy)
        else:
            raise TypeError("Invalid argument type!")

    """ This is the number of chebyshev polynomials stored in this class """
    @property
    def m(self):
        if len(self.shape) == 1:
            return 1
        return self.shape[1]

    """ Return the points at which the chebtech is sampled at """
    @property
    def x(self):
        return chebpts_type2_compute(self.n)

    @property
    def values(self):
        return polyval(self.coeffs)

    def happy(self, eps=None):
        eps = self.eps if eps is None else eps
        tol = np.maximum(self.hscale, self.vscale) * eps * np.ones(self.m)
        self.ishappy, cutoff = happiness_check(self.coeffs, tol)
        # cutoff is the last entry to keep -> increment by one.
        return bool(self.ishappy), cutoff+1

    def chebpts(self, n=None):
        if n is None: n = self.n
        return chebpts_type2_compute(self.n)

    def get_values(self):
        return polyval(self.coeffs)

    """ Construct a chebtechh from a callable op """
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

    """ Evaluates chebtech at x = -1 """
    def lval(self):
        c = np.copy(self.coeffs)
        c[1::2] *= -1
        return np.sum(c, axis=0)

    """ Evaluate chebtech at x = 1 """
    def rval(self):
        return np.sum(self.coeffs, axis=0)

    """ TODO: add support to get other kind of coefficients """
    def chebcoeff(self):
        return self.coeffs

    def restrict(self, s):
        """ Restrict the chebtech to a subinterval s """
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
        return chebtech(coeffs=coeffs, eps=self.eps, simplify=False,
                       maxLength=self.maxLength, ishappy=self.ishappy,
                       hscale=self.hscale)

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
        IF other is a ndarray: -> Evaluate the chebtechh at those points.
        IF other is a chebtechh: -> Compose the chebtechh
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
        chebtechh at the points y.
        """
        x, _, v, _ = self.points()
        # construct
        return barymat(y, x, v)

    def flipud(self):
        """ Flip / reverse a chebtech object such that G(x) = F(-x) for all x in [-1, 1] """
        coeffs = np.copy(self.coeffs)
        coeffs[1::2] *= -1
        return chebtech(coeffs=coeffs, eps=self.eps, hscale=self.hscale,
                       simplify=False, ishappy=self.ishappy,
                       maxLength=self.maxLength)

    def fliplr(self):
        """ Flip columns of an array-valued chebtech object. """
        return chebtech(coeffs=np.fliplr(self.coeffs), eps=self.eps,
                       hscale=self.hscale, simplify=False,
                       ishappy=self.ishappy, maxLength=self.maxLength)

    def __eq__(self, other):
        return np.all(self.shape == other.shape) and np.all(self.coeffs == other.coeffs)

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
        return roots(self, eps=self.eps)

    def minandmax(self, *args, **kwargs):
        fp = np.diff(self)
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
        Q, R = qr(self, *args, **kwargs)
        Q = chebtech(coeffs=Q, simplify=False, eps=self.eps, hscale=self.hscale,
                    maxLength=self.maxLength, ishappy=self.ishappy)
        return Q, R

    """ I/O support """
    def writeHDF5(self, fh):
        # Write the coefficients
        fd = fh.create_dataset('poly', data=self.coeffs)
        fd.attrs["eps"] = self.eps
        fd.attrs["hscale"] = self.hscale
        fd.attrs["maxLength"] = self.maxLength
        fd.attrs["ishappy"] = self.ishappy


def compose(f, op, g=None):
    if g is None:
        resampler = RefineCompose1(f, op)
        return chebtech(op=resampler)
    else:
        resampler = RefineCompose2(f, op, g)
        return chebtech(op=resampler)


def implements(np_function):
    """ Register an __array_function__ implementation """
    def decorator(func):
        HANDLED_FUNCTIONS[np_function] = func
        return func
    return decorator


@implements(np.argmax)
def argmax(f):
    return np.argmax(f.coeffs)


@implements(np.real)
def real(cheb):
    """ Returns real part of a chebtech """
    return chebtech(coeffs=np.real(cheb.coeffs), simplify=False,
                   ishappy=cheb.ishappy, eps=cheb.eps,
                   hscale=cheb.hscale, maxLength=cheb.maxLength)


@implements(np.imag)
def imag(cheb):
    """ Returns real part of a chebtech """
    return chebtech(coeffs=np.imag(cheb.coeffs), simplify=False,
                   ishappy=cheb.ishappy, eps=cheb.eps,
                   hscale=cheb.hscale, maxLength=cheb.maxLength)


@implements(np.conj)
def conj(cheb):
    return chebtech(coeffs=np.conj(cheb.coeffs), simplify=False, eps=cheb.eps,
                   ishappy=cheb.ishappy, maxLength=cheb.maxLength,
                   hscale=cheb.hscale)


@implements(np.diff)
def diff(cheb, n=1, axis=0):
    """ Compute the k-th derivative of the chebtech f """
    assert axis == 0, 'Axis other than zero not implemented yet!'

    # Simplify the coefficients prior to differentiating
    # Otherwise it seems errors may be accumulating.
    c = simplify_coeffs(cheb.coeffs, eps=cheb.eps)

    # Get the size of the current shape
    k = c.shape[0]

    # return zero if differentiating too much
    if n >= k:
        return chebtech(coeffs=np.zeros_like(cheb.coeffs), eps=cheb.eps,
                       hscale=cheb.hscale, simplify=False,
                       ishappy=cheb.ishappy, maxLength=cheb.maxLength)

    # Iteratively compute the coefficients of the derivatives
    c = computeDerCoeffs(c)
    for _ in range(1, n):
        c = computeDerCoeffs(c)

    # This returns a chebtech that has only N - k Chebyshev coefficients
    # -> note that setting happy will cause the function to be simplified!
    return chebtech(coeffs=c, ishappy=cheb.ishappy, simplify=False,
                   eps=cheb.eps, hscale=cheb.hscale,
                   maxLength=cheb.maxLength)


@implements(np.sum)
def sum(cheb, axis=0, **kwargs):
    """ Definite integral of a chebtech f on the interval [-1, 1].

    If f is an array-valued chebtech, then the result is a row vector
    containing the definite integrals of each column.

    """
    n = cheb.n

    # Constant cheb function
    if n == 1:
        return 2 * cheb.coeffs

    # Evaluate the integral by using the Chebyshev coefficients
    #
    # Int_{-1}^{1} T_k(x) dx = 2 / (1 - k^2)   if k even
    # Int_{-1}^{1} T_k(x) dx = 0               if k odd
    #
    # Thm 19.2 in Trefethen
    mask = np.ones_like(cheb.coeffs)
    mask[1::2, :] = 0
    out = np.expand_dims(np.hstack((2, 0, 2 / (1 - np.arange(2, n)**2))), axis=0) @ (cheb.coeffs * mask)
    return out.squeeze()


@implements(np.cumsum)
def cumsum(cheb, **kwargs):
    """ Indefinite integral of chebtech f, with the constant of integration
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

        TODO: FIXME!!!
    """
    n, m = cheb.shape
    c = np.vstack((cheb.coeffs, np.zeros((2, m))))  # pad with zeros
    b = np.zeros((n+1, m))

    # compute b_(2) ... b_(n+1)
    b[2:n+1, :] = (c[1:n, :] - c[3:n+2, :]) / np.tile(np.expand_dims(2*np.arange(2, n+1), axis=1), (1, m))
    # compute b(1)
    b[1, :] = c[0, :] - c[2, :] / 2
    v = np.ones((1, n))
    v[:, 1::2] = -1
    b[0, :] = np.matmul(v, b[1:, :])

    # Create the new chebtech
    g = chebtech(coeffs=b, eps=cheb.eps, simplify=False,
                hscale=cheb.hscale, ishappy=cheb.ishappy,
                maxLength=cheb.maxLength)

    # simplify
    g = g.simplify()

    # ensure that f(-1) = 0
    lval = g.lval()
    g.coeffs[0, :] = g.coeffs[0, :] - lval
    return g


@implements(np.inner)
def inner(cheb1, cheb2, weighted=False):
    """ Computes the L2 inner product on [-1, 1] of two Chebyshev series """
    n = len(cheb1) + len(cheb2)

    fvalues = polyval(prolong(cheb1.coeffs, n))
    gvalues = polyval(prolong(cheb2.coeffs, n))

    # compute Clenshaw-Curtis quadrature weights
    w = quadwts(n)

    # compute the inner-product
    out = np.matmul(fvalues.T * w, gvalues)

    # force non-negative output if the inputs are equal
    if cheb1 == cheb2:
        dout = np.diag(np.diag(out))
        out = out - dout + np.abs(dout)

    return out.squeeze()


def innerw(cheb1, cheb2):
    """
    Computes the weighted L2 inner product with the weight

        w(x) = 1 / sqrt(1 - x^2).

    This ensures that the Chebyshev functions of zeroth kind form
    an orthonormal basis, which simplifies the computation of the
    inner product.

                 /  0,     if n ِ≠ m      \
    (T_k, T_m) = |  π,     if n = m = 0  |
                 \  π / 2, if n = m ≠ 0  /

    """
    n = min(len(cheb1), len(cheb2))
    c1 = np.asarray(prolong(cheb1.coeffs, n))
    c2 = np.asarray(prolong(cheb2.coeffs, n))

    out = c1[0, :] * c2[0, :]
    out += np.sum(np.multiply(c1, c2), axis=0)
    out *= 0.5 * np.pi
    return out.squeeze()


@implements(np.dot)
def dot(cheb1, cheb2):
    rval = np.inner(cheb1, cheb2)
    return np.sum(np.diagonal(rval)).item() if rval.size > 1 else rval


@implements(np.hstack)
def hstack(chebs):
    n = np.max([len(cheb) for cheb in chebs])
    m = np.sum([cheb.size for cheb in chebs])

    # Create new coefficient storage
    coeffs = np.zeros((n, m), dtype=float, order='F')

    i = 0
    for cheb in chebs:
        sz = cheb.size
        coeffs[:, i:i+sz] = np.asarray(prolong(cheb.coeffs, n), order='F')
        i += sz

    maxLength = max([cheb.maxLength for cheb in chebs])
    eps       = max([cheb.eps for cheb in chebs])
    hscale    = max([cheb.hscale for cheb in chebs])   # TODO this is probably not correct!
    ishappy   = all([cheb.ishappy for cheb in chebs])

    return chebtech(coeffs=coeffs, ishappy=ishappy, simplify=False,
                   eps=eps, hscale=hscale,
                   maxLength=maxLength)


@implements(np.copy)
def copy(cheb):
    return chebtech(coeffs=np.copy(cheb.coeffs), ishappy=cheb.ishappy,
                   simplify=False, eps=cheb.eps, hscale=cheb.hscale,
                   maxLength=cheb.maxLength)


# @implements(np.hsplit)
# def hsplit(cheb):
#     return [chebtech(coeffs=cheb.coeffs[:, i], ishappy=cheb.ishappy,
#                     simplify=True, eps=cheb.eps) for i in range(cheb.size)]
