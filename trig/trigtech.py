#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
import scipy.linalg as LA
from copy import deepcopy
from numbers import Number
from math import floor, ceil
from scipy.fft import ifft, fft

from funpy.cheb.detail import standardChopCmplx
from funpy.cheb.chebpy import chebtec
from funpy.trig.trig_simplify import simplify_coeffs
from funpy.cheb.refine import FunctionContainer
from funpy.trig.refine import Refine, RefineBase
from funpy.trig.eval import horner
from funpy.trig.transform import coeffs2vals, vals2coeffs
from funpy.trig.trigpts import trigpts, quadwts
from funpy.trig.trig_simplify import prolong


""" Do the inheritance correctly otherwise we have all kinds of isinstance problems """
class trigtech(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, op=None, file=None, values=None, coeffs=None, *args, **kwargs):
        self.maxLength = 4096

        self.eps = kwargs.pop('eps', 1e-10)
        self.hscale = kwargs.pop('hscale', self.eps)
        self.vscale = kwargs.pop('vscale', self.eps)

        # Am I happy?
        self.ishappy = kwargs.pop('ishappy', False)
        self.isReal = kwargs.pop('isReal', False)

        # This function keeps both the values and coeffs
        self.coeffs = np.zeros((0, 0), order='F', dtype='complex')
        self.values = np.zeros((0, 0), order='F', dtype='complex')

        if file is not None:
            raise NotImplemented
        elif op is not None:
            if isinstance(op, np.ndarray):
                if len(op.shape)==1:
                    op = np.expand_dims(op, axis=1)
                self.values = op.astype(complex)
                self.coeffs = vals2coeffs(op)

                # we are always happy
                self.ishappy = True

                # Threshold to determine if f is real or not
                vscale = np.maximum(self.vscale, self.get_vscale())
                self.isReal = self.get_isreal(vscale)
                self.values[:, self.isReal] = np.real(self.values[:, self.isReal])
            else:
                # Create callable container
                if isinstance(op, list):
                    op = FunctionContainer(op, dtype=np.complex128)
                # check what kind of op we have
                if isinstance(op, RefineBase):
                    self.populate(op)
                else:   # Assume that it's a lambda -> TODO check that
                    # TODO implement this for the other parts too
                    self.__check_callable(op)
                    refine = Refine(op=op, strategy=kwargs.pop('resample', 'nested'))
                    self.populate(refine)

                # check whether the function should be real or not ???
                vscale = np.maximum(self.vscale, self.get_vscale())
                self.isReal = self.get_isreal(vscale)

                # call simplify
                self.simplify()

        elif values is not None or coeffs is not None:
            if values is not None:
                self.values = values.astype(complex)

            if coeffs is not None:
                self.coeffs = coeffs.astype(complex)

            # If one is not set compute the other
            if self.values.size == 0:
                self.values = coeffs2vals(self.coeffs)

            if self.coeffs.size == 0:
                self.coeffs = vals2coeffs(self.values)

            # we are always happy
            self.ishappy = True

            # Threshold to determine if f is real or not
            vscale = np.maximum(self.vscale, self.get_vscale())

            # Make sure the expected real values are real
            self.isReal = self.get_isreal(vscale)
            self.values[:, self.isReal] = np.real(self.values[:, self.isReal])
            # force complex dtype ? is there a better way?
            self.values = self.values.astype(complex)
        else:
            assert False, 'Don\'t know what to do with this!'

    def __deepcopy__(self, memo):
        id_self = id(self)
        _copy = memo.get(id_self)
        if _copy is None:
            _copy = type(self)(
                coeffs=deepcopy(self.coeffs),
                values=deepcopy(self.values),
                eps=self.eps, hscale=self.eps,
                maxLength=self.maxLength,
                ishappy=self.ishappy,
                isreal=self.isReal)
            memo[id_self] = _copy
        return _copy

    def __repr__(self):
        # TODO: COMPLETE ME
        return f"{self.__class__.__name__}(coeffs={self.coeffs})"

    def __str__(self):
        # TODO: FINISH ME
        return 'trigtech(coeffs={0:s})'.format(str(self.coeffs.flatten()))

    def __len__(self):
        return self.coeffs.shape[0]

    def __array__(self):
        """ Is called when object is passed to np.asarray or np.array """
        return self.coeffs

    def __array_ufunc__(self, numpy_ufunc, method, *inputs, **kwargs):
        import funpy.trig.ufuncs as cp_funcs
        out = kwargs.get('out', ())
        # for x in inputs + out:
        #     if not isinstance(x, (np.ndarray, Number, type(self))):
        #         return NotImplemented

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
                op = lambda x: numpy_ufunc(inputs[0](x))
                return trigtech(op=op)
            elif len(inputs) == 2:
                op = lambda x: numpy_ufunc(inputs[0](x), inputs[1](x))
                return trigtech(op=op)
            else:
                return NotImplemented
        else:
            return NotImplemented

    def isfortran(self):
        return self.coeffs.flags.f_contiguous

    def __len__(self):
        return self.coeffs.shape[0]

    @property
    def type(self):
        return 'trig'

    @property
    def ndim(self):
        return self.coeffs.ndim

    @property
    def size(self):
        return self.coeffs.shape[1]

    @property
    def shape(self):
        return self.coeffs.shape

    @property
    def n(self):
        return self.shape[0]

    @property
    def m(self):
        return self.shape[1]

    @property
    def istrig(self):
        return True

    @property
    def const_index(self):
        if np.remainder(self.n, 2):  # n odd
            const_index = (self.n+1)//2 - 1
        else:
            const_index = self.n//2

        return const_index

    """ Useful to select a column of function """
    def __getitem__(self, idx):
        assert idx >= 0 and idx < self.m, 'Index %d out of range [0, %d].' % (idx, self.m-1)
        if idx < 0 or idx >= self.m:
            raise IndexError

        fun = trigtech(coeffs=self.coeffs[:, None, idx],
                       values=self.values[:, None, idx], simplify=False,
                       ishappy=self.ishappy, isreal=self.isReal)

        return fun

    def __iter__(self):
        self.ipos = 0
        return self

    def __next__(self):
        if self.ipos >= self.m:
            raise StopIteration
        self.ipos += 1
        return self[self.ipos-1]

    def __construct_from_callable(self, callable):
        pass

    """ Return the points at which the trigtech is sampled at """
    @property
    def x(self):
        # TODO: avoid all the extra computations here!!
        x, _ = trigpts(self.n)
        return x

    @property
    def isreal(self):
        return np.all(self.isReal)

    def restrict(self, s):
        """ Restrict a trigtech to a subinterval """
        pass

    def get_isreal(self, vscl=1):
        return (np.max(np.abs(np.imag(self.values)), axis=0) <= 3 * self.eps * vscl).squeeze()

    def get_vscale(self):
        """ Estimate the vertical scale of a function """
        if self.coeffs.size == 0:
            return 0
        elif self.n == 1:
            return np.abs(self.coeffs)
        else:
            vals = coeffs2vals(self.coeffs)
            # TODO: Same again why does np.abs -> throw floating point errors?
            return np.max(np.hypot(np.real(vals), np.imag(vals)), axis=0)

    """ prolong """
    def prolong(self, Nout):
        # If Nout < length(self) -> compressed by chopping
        # If Nout > length(self) -> coefficients are padded by zero
        Nin = self.n

        if Nout == Nin:  # Do nothing
            return self

        coeffs = self.coeffs
        if np.remainder(Nin, 2) == 0:
            coeffs = np.vstack((0.5 * coeffs[0, :], coeffs[1:, :], 0.5 * coeffs[0, :]))
            Nin += 1

        if Nin == Nout:
            self.coeffs = coeffs
            self.values = coeffs2vals(self.coeffs)
            self.values[:, self.isReal] = np.real(self.values[:, self.isReal])

        # Pad with zeros
        if Nout > Nin:
            kup = np.ceil((Nout-Nin)/2).astype(int)
            kdown = np.floor((Nout-Nin)/2).astype(int)
            coeffs = np.vstack((np.zeros((kup, coeffs.shape[1])),
                                coeffs,
                                np.zeros((kdown, coeffs.shape[1]))))
            self.coeffs = coeffs
            self.values = coeffs2vals(self.coeffs)
            self.values[:, self.isReal] = np.real(self.values[:, self.isReal])

        # chop coefficients
        if Nout < Nin:
            kup = np.floor((Nin-Nout)/2).astype(int)
            kdown = np.ceil((Nin-Nout)/2).astype(int)
            coeffs = coeffs[kup:-kdown, :]
            if kup < kdown:
                coeffs[0, :] = 2*coeffs[0, :]
            self.coeffs = coeffs
            self.values = coeffs2vals(self.coeffs)
            self.values[:, self.isReal] = np.real(self.values[:, self.isReal])

        return self

    def happy(self):
        coeffs = np.abs(self.coeffs[::-1, :])
        n, m = coeffs.shape

        if np.remainder(n, 2) == 0:
            coeffs = np.vstack((coeffs[n-1, :],
                                coeffs[n-2:n//2-1:-1, :] + coeffs[:n//2-1, :],
                                coeffs[n//2-1, :]))
        else:
            coeffs = np.vstack((coeffs[n-1:(n+1)//2-1:-1, :] + coeffs[:(n+1)//2-1, :],
                                coeffs[(n+1)//2-1, :]))

        coeffs = np.flipud(coeffs)
        coeffs = np.vstack((coeffs[0, :], np.kron(coeffs[1:, :], np.vstack((1,1)))))

        tol = np.max(self.eps) * np.ones(m)
        # TODO: FIX THIS!
        #values = polyval(self.coeffs)
        #vscaleF = np.max(np.abs(values), axis=0)
        #tol = tol * np.maximum(self.hscale, self.vscale / vscaleF)
        ishappy = np.zeros(m).astype(bool)
        cutoff = np.zeros(m).astype(int)
        for k in range(m):
            # FIXME!
            cutoff[k] = standardChopCmplx(np.asfortranarray(self.coeffs[:, k]), tol[k])
            # check if happy - TODO: check this for numpy adaptation
            ishappy[k] = (cutoff[k] < n)

            if np.remainder(cutoff[k], 2) == 0:
                cutoff[k] = cutoff[k]//2
            else:
                cutoff[k] = (cutoff[k]-1)//2

            # exit if any column is unhappy
            if not ishappy[k]:
                break

        ishappy = np.all(ishappy)
        cutoff  = 2*np.max(cutoff)+1
        self.ishappy = ishappy
        return ishappy, cutoff

    """ check whether op produces valid results """
    def __check_callable(self, callable):
        # check at a random point to check whether result is complex and if
        # values is a NaN or Inf
        pseudoRand = 0.376989633393435
        rndVal = callable(np.asarray(2 * pseudoRand - 1)).squeeze()

        if np.any(np.isnan(rndVal)) or np.any(np.isinf(rndVal)):
            assert False, 'Cannot handle functions that evaluate to Inf or NaN'

        self.isReal = np.zeros(rndVal.size).astype(bool)
        if not rndVal.shape:
            self.isReal[0] = np.isreal(rndVal)
        else:
            self.isReal = np.zeros_like(rndVal).astype(bool)
            for k, rval in enumerate(rndVal):
                self.isReal[k] = np.isreal(rval)

    """ Construct a chebtech from a callable op """
    def populate(self, refine):
        while True:
            self.values, giveUp = refine(self)
            if giveUp:
                break

            # update vscale
            valuesTemp = np.copy(self.values)
            valuesTemp[~np.isfinite(self.values)] = 0
            # TODO: Again why?
            self.vscale = max(self.vscale, np.max(np.hypot(np.real(valuesTemp), np.imag(valuesTemp))))

            # compute coefficients
            self.coeffs = vals2coeffs(self.values)

            # check happiness
            ishappy, cutoff = self.happy()

            if ishappy:
                self.prolong(cutoff)
                break

        # update real information
        self.ishappy = ishappy
        self.values[:, self.isReal] = np.real(self.values[:, self.isReal])
        # Always force the dtype of the array to be complex!
        self.values = self.values.astype(complex)

    def simplify(self, eps=None, tol=None):
        # if not happy simply do nothing
        if not self.ishappy:
            return self

        if eps is None: eps = self.eps

        # Call the simplify coefficients
        self.values, self.coeffs = simplify_coeffs(self.coeffs, self.isReal, eps=eps)
        return self

    def __call__(self, other):
        return self.feval(other)

    def feval(self, other):
        return self.horner(other)

    def horner(self, x):
        x = np.atleast_1d(x)
        return horner(x, self.coeffs, self.isReal).squeeze()

    def diff(self, k=1, axis=0):
        """ Computes the derivative of a trigtech """
        assert axis == 0, 'other not implemented'
        n = self.n
        c = np.copy(self.coeffs)

        if n & 1:
            waveNumber = np.expand_dims(np.arange(-(n-1)/2, n/2), axis=1)
        else:
            waveNumber = np.expand_dims(np.arange(-n/2, n/2), axis=1)

        # derivative in Fourier space
        c = c * (1j * np.pi * waveNumber)**k
        return trigtech(coeffs=c, ishappy=self.ishappy, isreal=self.isReal, simplify=False)

    def diff_fast(self, k=1, axis=0):
        """ Computes the derivative of a trigtech Fourier space """
        assert axis == 0, 'other not implemented'
        n = self.n

        if n & 1:
            waveNumber = np.expand_dims(np.arange(-(n-1)/2, n/2), axis=1)
        else:
            waveNumber = np.expand_dims(np.arange(-n/2, n/2), axis=1)

        # derivative in Fourier space
        return self.coeffs * (1j * np.pi * waveNumber)**k

    def diffmat(self, k=1):
        """ Trigonometric differentiation matrix

        maps functions values at N equispaced grid points in [0, 2*pi) to
        values of the derivative of the Fourier interpolant at those points.

        This should be moved somewhere else!
        """
        n = self.n
        h = 2*np.pi/n

        # sign
        sgn = np.expand_dims((-1)**(np.arange(1, n)), axis=1)

        # indices
        n1 = floor((n-1)/2)
        n2 = ceil((n-1)/2)

        # grid points on [-1, 0)
        v = np.expand_dims(np.arange(1, n2+1), axis=1) * h/2

        if k == 0:
            return np.eye(n)
        elif k == 1:
            if n & 1:
                tmp = 1./np.sin(v)
                col = np.vstack((0, (np.pi/2)*sgn*np.vstack((tmp, np.flipud(tmp[:n1])))))
            else:
                tmp = 1./np.tan(v)
                col = np.vstack((0, (np.pi/2)*sgn*np.vstack((tmp, -np.flipud(tmp[:n1])))))

            # form the first row
            row = -col

        elif k == 2:
            # form columns by flipping trick
            if n & 1:
                tmp = (1./np.sin(v)) * (1./np.tan(v))
                col = np.pi**2 * np.vstack((-np.pi**2/(3*h**2)+1/12, -0.5 * sgn * np.vstack((tmp, -np.flipud(tmp[:n1])))))
            else:
                tmp = (1./np.sin(v))**2
                col = np.pi**2 * np.vstack((-np.pi**2/(3*h**2)-1/6, -0.5 * sgn * np.vstack((tmp, np.flipud(tmp[:n1])))))

            # for the first row
            row = col
        else:
            # Form the first column using fft
            n3 = (-n/2)*np.remainder(k+1,1)*np.ones(np.remainder(n+1,2))
            waveNumber = 1j*np.hstack((np.arange(n1+1), n3, np.arange(-n1,0)))
            col = np.pi**k * np.real(ifft(waveNumber**k * fft(np.eye(1,n))))

            # for the first row
            if k & 1:
                col = np.hstack((0, col[0, 1:]))
                row = -col
            else:
                row = col

        # form the differentiation matrix which is toeplitz
        return LA.toeplitz(col, row)

    """ Definite Integral """
    def sum(self, axis=None, **unused_kwargs):
        if axis is None:
            axis = 0
        assert axis == 0, ''
        n = self.n
        out = 2 * self.coeffs[None, floor((n+2)/2)-1, :]
        out[:, self.isReal] = np.real(out[:, self.isReal])
        return out.squeeze()

    """ Indefinite Integral """
    def cumsum(self, m=1, **unused_kwargs):
        """ Indefinite integral of a trigtech F, whose mean is zero, with the constant of
        integration chosen such as F(-1) = 0. If the mean is not zero, the result would no longer
        be periodic thus an error is thrown.

        If the trigtech of length n is represented by the truncated series

            sum_{k = -(n-1)/2}^{(n-1)/2} c_k exp(i*pi*kx)

        its integral is represented with a trigtech of length n given by

            sum_{k = -(n-1)/2}^{(n-1)/2} b_k exp(i*pi*kx)

        where b_0 is determined from the constant of integration as

            b_0 = sum_{k=-(n-1)/2}^{(n-1)/2} (-1)^k / (i pi k) c_k

        with c_0 = 0. The other coefficients are given by

            b_k = c_k / (i pi k).

        """
        c = self.coeffs
        n = self.n
        isEven = np.remainder(n, 2) == 0

        # Index of the constant coefficients
        if isEven:
            ind = n//2 + 1
        else:
            ind = (n+1)//2

        # check that the mean of the trigtech is zero. If it is not, then throw an error.
        if np.any(np.abs(self.coeffs[ind, :])) > 1e1*self.vscale*self.eps:
            raise RuntimeError("Indefinite integrals are only possible for trigtech objects with zero mean!")

        # throw error that this is only possible for mean zero trigtechs
        if isEven:
            # set coeff corresponding to the constant mode to zero:
            c[n//2, :] = 0
            # expand the coefficients to be symmetric (see above discussion)
            c[0, :] = 0.5 * c[0, :]
            c = np.vstack((c, c[0, :]))
            highestDegree = n//2
        else:
            c[(n+1)//2, :] = 0
            highestDegree = (n-1)//2

        # loop over integration factor for each coefficient
        sumIndices = np.expand_dims(np.arange(-highestDegree, highestDegree+1), axis=1)
        integrationFactor = (-1j/sumIndices/np.pi)**m
        # zero out the one corresponding to the zeroth term
        integrationFactor[highestDegree] = 0
        c = c * integrationFactor
        # If this is an odd order cumsum and there an even number of
        # coefficients then zero out the coefficients corresponding to
        # sin(N/2x) term, since this will be zero on the Fourier grid
        if m & 1 and isEven:
            c[0, :] = 0
            c[n, :] = 0

        # fix the constant term
        c[highestDegree+1, :] = -np.sum(c * (-1+0j)**sumIndices)

        if isEven:
            c = c[:-1]

        # call simplify
        # grab lval

        return trigtech(coeffs=c, simplify=False, ishappy=self.ishappy, isreal=self.isReal)

    def innerproduct(self, other):
        n = len(self) + len(other)

        # Get the values
        _, fvalues = prolong(self.coeffs, n, self.isReal)
        _, gvalues = prolong(other.coeffs, n, other.isReal)

        # Compute the quadrature weights
        w = quadwts(n)

        # Compute the inner product
        # Inner product in a Hilbert space is (f, conj(g))
        out = np.matmul(fvalues.T * w, np.conj(gvalues))

        # FIXME for array valued trigtechs!
        if np.all(self.isReal * other.isReal):
            out = np.real(out)

        # Force non-negative output when the inputs are the same
        if self == other:
            pass

        return out.squeeze()

    def real(self):
        """ Returns real part of a trigtech """
        if self.isreal:
            return self

        fvalues = np.real(self.values)
        if np.all(fvalues < self.eps):
            nisReal = np.ones(self.m).astype(bool)
            return trigtech(values=np.zeros_like(self.values),
                            coeffs=np.zeros_like(self.coeffs),
                            ishappy=self.ishappy, simplify=False,
                            isreal=nisReal)
        else:
            ncoeffs = vals2coeffs(fvalues)
            nisReal = np.ones(self.m).astype(bool)

        return trigtech(values=fvalues, coeffs=ncoeffs, ishappy=self.ishappy,
                        simplify=False, isreal=nisReal)

    def imag(self):
        """ Returns imag part of a trigtech """
        if self.isreal:
            return trigtech(values=np.zeros_like(self.values),
                            coeffs=np.zeros_like(self.coeffs),
                            ishappy=self.ishappy, simplify=False,
                            isreal=True)
        else:
            # compute the imaginary part
            nvalues = np.imag(self.values)
            ncoeffs = vals2coeffs(nvalues)
            nisReal = np.ones(self.m).astype(bool)

            return trigtech(values=nvalues, coeffs=ncoeffs,
                            ishappy=self.ishappy, simplify=False,
                            isreal=nisReal)

    def conj(self):
        id = ~self.isReal
        if np.all(id):
            return self

        # other wise construct a new trigtech
        values = np.copy(self.values)
        coeffs = np.copy(self.coeffs)
        values[:, id] = np.conj(values[:, id])
        coeffs[:, id] = np.flipud(np.conj(coeffs[:, id]))
        return trigtech(values=values, coeffs=coeffs, ishappy=self.ishappy,
                        simplify=False, isreal=self.isReal)

    def minandmax(self, *args, **kwargs):
        g = chebtec(op=lambda x: self.feval(x))
        return g.minandmax(*args, **kwargs)

    def compose(self, op, g=None):
        """ Returns a lambda generating the composition of the two functions """
        if g is None:
            """ Compute op(f) """
            return lambda x: op(self(x))
        else:
            """ Compute op(f, g) """
            return lambda x: op(self(x), g(x))

    def truncate(self, new_size):
        """ Return the truncated coefficients of this function """
        const_index = self.const_index
        k = (new_size // 2)
        c = self.coeffs
        return np.vstack((c[const_index-k:const_index],
                          c[const_index:const_index+k]))

    # TODO -> somehow reduce the endless code duplication of this code-path!
    def prolong_coeffs(self, Nout):
        # If Nout < length(self) -> compressed by chopping
        # If Nout > length(self) -> coefficients are padded by zero
        Nin = self.coeffs.shape[0]

        if np.remainder(Nin, 2) == 0:
            c = np.vstack((0.5 * self.coeffs[0, :], self.coeffs[1:, :], 0.5 * self.coeffs[0, :]))
            Nin += 1
        else:
            c = np.copy(self.coeffs, order='K')

        if Nin == Nout:
            c = self.coeffs

        # Pad with zeros
        elif Nout > Nin:
            kup = np.ceil((Nout-Nin)/2).astype(int)
            kdown = np.floor((Nout-Nin)/2).astype(int)
            c = np.vstack((np.zeros((kup, c.shape[1])), c,
                           np.zeros((kdown, c.shape[1]))))

        # chop coefficients
        elif Nout < Nin:
            kup = np.floor((Nin-Nout)/2).astype(int)
            kdown = np.ceil((Nin-Nout)/2).astype(int)
            c = c[kup:-kdown, :]
            if kup < kdown:
                c[0, :] = 2*c[0, :]

        else:
            raise RuntimeError("I should not get here!")

        return c

    def __eq__(self, other):
        return np.all(self.shape == other.shape) and \
                np.all(self.coeffs == other.coeffs) and \
                np.all(self.values == other.values)


def trigpoly(n, interval=[-1, 1]):
    # TODO: Deal with other kinds of polynomials
    assert not np.any(np.isinf(interval)), 'Can\'t deal with infinite domains.'
    n = np.asarray(n).astype(int)

    # construct Chebyshev coefficients
    N = np.max(n) + 1
    c = np.eye(N)
    c = c[:, n]

    # construct the polynomial
    return trigtech(coeffs=c)
