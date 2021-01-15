#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
import h5py as h5
from numbers import Number
from copy import deepcopy, copy
from funpy.support.cached_property import lazy_property
from funpy.trig.trigtech import trigtech
from funpy.trig.operations import circconv, circshift, trig_adhesion
from funpy.cheb.chebpy import chebtec
from funpy.mapping import Mapping

HANDLED_FUNCTIONS = {}
SMALL_EPS = 1e-8

class Fun(np.lib.mixins.NDArrayOperatorsMixin):
    r"""
        Wrapper for bounded functions on [a, b]
    """
    def __init__(self, file=None, *args, **kwargs):
        # the approximation on [-1, 1] - needs to be an array for piece-wise defined functions
        self.onefun = kwargs.pop('onefun', None)

        # grab operator
        op = kwargs.pop('op', None)
        compose = kwargs.pop('compose', True)

        # if file is not None create function from None
        if file is not None:
            self.readHDF5(file)
        else:
            # When the domain has length > 2 then we have a piece-wise defined function
            self.domain = kwargs.pop('domain', np.asanyarray([-1, 1]))

        # mapping for [a, b] -> [-1, 1]
        self.mapping = kwargs.pop('mapping', Mapping(ends=self.domain))

        if compose and (op is not None and (callable(op) or np.all([callable(o) for o in op]))):
            def get_op(op):
                # This will get its own scope so op won't be overwritten!
                def f(x):
                    return op(self.mapping.fwd(x))
                return f

            op = op if isinstance(op, list) else list([op])
            orig_op = deepcopy(op)
            op = [get_op(oop) for oop in orig_op]

        # Note that we still have to turn lone compose lambdas into a list!
        elif op is not None and (callable(op) or all([callable(o) for o in op])):
            op = op if isinstance(op, list) else list([op])

        # construct the function
        if self.onefun is None:
            self.__construct(op=op, hscale=self.hscale, *args, **kwargs)

    def __construct(self, *args, **kwargs):
        fun_type = kwargs.pop('type', 'cheb')
        if fun_type == 'trig':
            self.onefun = trigtech(*args, **kwargs)
        else:  # we will simply default to cheb
            self.onefun = chebtec(*args, **kwargs)

    @property
    def istrig(self):
        return self.onefun.istrig

    @property
    def type(self):
        if self.istrig:
            return 'trig'
        else:
            return 'cheb'

    @property
    def ndim(self):
        return self.onefun.ndim

    """ Return the points at which the chebtec is sampled at """
    @property
    def x(self):
        return self.mapping.fwd(self.onefun.x)

    @lazy_property
    def hscale(self):
        return np.linalg.norm(self.domain, np.inf)

    def simplify(self, *args, **kwargs):
        self.onefun = self.onefun.simplify(*args, **kwargs)
        return self

    def prolong(self, Nout):
        self.onefun = self.onefun.prolong(Nout)
        return self

    def norm(self, p=2):
        return norm(self, p=p)

    def __len__(self):
        return len(self.onefun)

    def __str__(self):
        return '%sfun column (%d pieces) on %s at %s points.' % (self.type, self.m, self.domain, len(self))

    def __repr__(self):
        # TODO: COMPLETE ME
        return f"{self.type}{self.__class__.__name__}(domain={self.domain}, mapping={self.mapping}, onefun={self.onefun})"

    def __getattr__(self, name):
        if not hasattr(self.onefun, name):
            raise AttributeError(name)
        return getattr(self.onefun, name)

    def __call__(self, x):
        """ x in [a, b] -> [-1, 1] """
        z = self.mapping.bwd(x)
        assert np.min(z) >= -1.0 and np.max(z) <= 1.0, 'Mapping must return an interval [-1, 1] instead of [%.12g, %.12g]!' % (np.min(z), np.max(z))
        return self.onefun.feval(z)

    """ Implement array ufunc support """
    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    """ Shallow copy handler """
    def __copy__(self):
        return type(self)(domain=self.domain, mapping=self.mapping,
                          onefun=self.onefun)

    def __getitem__(self, idx):
        # TODO: this is not efficient!
        try:
            new_fun = self.onefun[idx]
        except Exception as e:
            raise e
        return Fun(domain=self.domain, mapping=self.mapping, onefun=new_fun)

    def __iter__(self):
        self.ipos = 0
        return self

    def __next__(self):
        if self.ipos >= self.m:
            raise StopIteration
        self.ipos += 1
        return self[self.ipos-1]

    def __array_ufunc__(self, numpy_ufunc, method, *inputs, **kwargs):
        out = kwargs.get('out', ())
        for x in inputs + out:
            # if not isinstance(x, (np.ndarray, Number, type(self))):
            #     return NotImplemented

            # check domain
            if isinstance(x, type(self)):
                if not np.all(np.asarray(x.domain) - np.asarray(self.domain) < SMALL_EPS):
                    raise ValueError("Domain mismatch %s != %s!" % (x.domain, self.domain))

        if method == "__call__":
            out = kwargs.pop('out', None)
            ipts = [x.onefun if isinstance(x, type(self)) else x for x in inputs]
            if out is not None:
                numpy_ufunc(*ipts, out=out[0].onefun, **kwargs)
                return out[0]
            else:
                new_fun = numpy_ufunc(*ipts, **kwargs)
                return Fun(domain=self.domain, mapping=self.mapping, onefun=new_fun)
        else:
            return NotImplemented

    """ Deep copy handler """
    def __deepcopy__(self, memo):
        id_self = id(self)
        _copy = memo.get(id_self)
        if _copy is None:
            _copy = type(self)(
                domain=deepcopy(self.domain, memo),
                mapping=deepcopy(self.mapping, memo),
                onefun=deepcopy(self.onefun, memo))
            memo[id_self] = _copy
        return _copy

    """ I/O support """
    def writeHDF5(self, fh, grp_name='fun'):
        grp = fh.create_group(grp_name)
        self.onefun.writeHDF5(grp)
        grp.attrs['domain'] = self.domain

    def readHDF5(self, fh, grp_name='fun', *args, **kwargs):
        fh = fh[grp_name]
        self.domain = fh.attrs['domain']
        self.__construct(file=fh['poly'], *args, **kwargs)

def implements(np_function):
    """ Register an __array_function__ implementation """
    def decorator(func):
        HANDLED_FUNCTIONS[np_function] = func
        return func
    return decorator


def zeros(m, domain=[-1, 1], type='cheb'):
    return Fun(domain=domain, type=type, op=np.zeros((1, m)), simplify=False)

@implements(np.zeros_like)
def zeros_like(f):
    return Fun(domain=f.domain, mapping=f.mapping, type=f.type,
               op=np.zeros((f.n, f.m), order='F'), simplify=False)

def ones(m, domain=[-1, 1], type='cheb'):
    return Fun(domain=domain, type=type, op=np.ones((1, m)), simplify=False)

@implements(np.ones_like)
def ones_like(f):
    return Fun(domain=f.domain, mapping=f.mapping, type=f.type,
               op=np.ones((f.n, f.m), order='F'), simplify=False)

@implements(np.real)
def real(f):
    if f.istrig:
        nf = f.onefun.real()
        return Fun(domain=f.domain, mapping=f.mapping, onefun=nf)

    # otherwise it's a dummy!
    return f

@implements(np.imag)
def imag(f):
    if f.istrig:
        nf = f.onefun.imag()
        return Fun(domain=f.domain, mapping=f.mapping, onefun=nf)

    # otherwise it's a dummy!
    return zeros_like(f)

def zeros(n, domain=[-1, 1], *args, **kwargs):
    """ Creates n functions equally zero. """
    return Fun(coeffs=np.zeros((1, n), dtype=np.float, order='F'), domain=domain, *args, **kwargs)

def ones(n, domain=[-1, 1], *args, **kwargs):
    """ Creates n functions equally zero. """
    return Fun(coeffs=np.ones((1, n), dtype=np.float, order='F'), domain=domain, *args, **kwargs)

def asfun(obj, domain=[-1, 1], *args, **kwargs):
    if isinstance(obj, Fun):
        return obj
    elif isinstance(obj, Number) or obj.size == 1:
        return Fun(coeffs=np.reshape(np.asarray(obj), (1, 1), order='F'), domain=domain, *args, **kwargs)
    assert False, 'Unsupported type %s!' % type(obj)

@implements(np.dot)
def dot(matrix, function):
    return np.dot(matrix, function.values)

def get_composer(f, op, g=None):
    """ Returns a lambda generating the composition of the two functions """
    if g is None:
        """ Compute op(f) """
        return lambda x: op(f(x))
    else:
        """ Compute op(f, g) """
        return lambda x: op(f(x), g(x))

def compose(f, op, g=None, ftype=None):
    cop = get_composer(f, op, g)
    # if target type is the same as before we just move on
    if ftype is None:
        ftype = f.type

    return Fun(domain=f.domain, mapping=f.mapping, type=ftype, op=cop,
               compose=False)

def qr(f, *args, **kwargs):
    """ Compute the QR decomposition of the Fun object f"""
    rescaleFactor = np.sqrt(0.5 * np.diff(f.domain))
    Q, R = f.onefun.qr(*args, **kwargs)
    Q /= rescaleFactor
    R *= rescaleFactor
    return Fun(domain=f.domain, mapping=f.mapping, type=f.type, onefun=Q), R

def norm(f, p=2):
    """ Computes the p-norm of f

        p: a positive integer.

        If f has different columns it assumes that we are dealing with a Cartesian product space
        and sums the p-norms prior to taking the p-th root.
    """
    if p <= 0:
        raise ValueError("The norm index must be positive!")

    def detail(f, p):
        # TODO: why do I need to cast to real here?
        if p == 1:
            return np.sum(np.sum(abs(f)))
        elif p == np.inf:
            vals, pos = minandmax(f)
            return np.max(np.abs(vals))
        elif p & 1:
            return np.sum(np.sum(abs(f)**p))
        else:
            # Take absolute value of the integral to avoid small negative solutions!
            # TODO: this should be fixed somewhere else!
            #print('sum = ', np.real(np.sum(f**p)))
            return np.abs(np.sum(np.sum(f**p)))

    if f.istrig:
        fR = np.real(f)
        fI = np.imag(f)
        if p == 1:
            return (detail(fR, p) + detail(fI, p)).squeeze()
        elif p == np.inf:
            return max(detail(fR, p), detail(fI, p)).squeeze()
        else:
            return np.power(detail(fR, p) + detail(fI, p), 1. / p).squeeze()
    else:
        norm_p = detail(f, p)
        if p == 1 or p == np.inf:
            return norm_p.squeeze()
        else:
            return np.power(norm_p, 1. / p).squeeze()

def norm2(f, p=2):
    """ Computes || f ||_p^p """
    if p <= 0:
        raise ValueError("The norm index must be positive!")

    def detail(f, p):
        # TODO: why do I need to cast to real here?
        if p == 1:
            return np.sum(np.real(np.sum(abs(f)))).squeeze()
        elif p & 1:
            return np.sum(np.real(np.sum(abs(f)**p))).squeeze()
        else:
            return np.sum(np.real(np.sum(f**p))).squeeze()

    if f.istrig:
        fR = np.real(f)
        fI = np.imag(f)
        return detail(fR, p) + detail(fI, p)
    else:
        return detail(f, p)

def h1norm(f, p=2, k=1):
    """ Compute the k, p Sobolev norm of f """
    if p <= 0 or k < 0:
        raise ValueError("The norm index must be positive!")

    norm = norm2(f, p=p)
    for i in range(1, k+1):
        f = np.diff(f)
        norm += norm2(f, p=p)
    return np.power(norm, 1./p)

def sturm_norm(f, p=2):
    """ Computes the p "Sturm" norm of the function f(x) defined on the interval [a, b]

            sign[f'(a)] * || f ||_p
    """
    df = np.diff(f)
    df_norm = norm(df, p=p)
    snorm = np.real(np.sign(df(df.domain[0]))) * norm(f, p=p) if df_norm > 1e-7 else norm(f, p=p)
    return snorm

def sturm_norm_alt(f, p=2):
    """ Computes the p "Sturm" norm of the function f(x) defined on the interval [a, b]

            sign[f(a)] * || f ||_p
    """
    df = np.diff(f)
    df_norm = norm(df, p=p)
    snorm = np.real(np.sign(f(f.domain[0]))) * norm(f, p=p) if df_norm > 1e-7 else norm(f, p=p)
    return snorm

@implements(np.convolve)
def convolve(f, g):
    assert f.istrig == g.istrig, 'Fun type must match for computing convolutions!'
    if f.istrig:
        # Compute a circular convolution
        if not np.all(f.domain == g.domain):
            raise ValueError("Domain mismatch %s != %s!" % (f.domain, g.domain))

        rescaleFactor = 0.5 * np.diff(f.domain)
        df = circconv(f, g, f.domain) * rescaleFactor
        return Fun(domain=f.domain, mapping=f.mapping, onefun=df)
    else:
        # Compute a traditional convolution
        return NotImplemented

def adhesion(f, g):
    """ Computes the adhesion operator

        K[u](x) = Int_{-1}^{1} f(x + r) g(r) dr

        g -> must be a trigtech containing the correctly scaled fourier coefficients of the
        integration kernel!

    """
    assert f.istrig == g.istrig, 'Fun type must match for computing convolutions!'
    if f.istrig:
        # Simply multiply the coefficients together!
        df = trig_adhesion(f, g, f.domain)
        return Fun(domain=f.domain, mapping=f.mapping, onefun=df)
    else:
        # Compute a traditional convolution
        return NotImplemented

@implements(np.conj)
def conj(f):
    new_fun = f.onefun.conj()
    return Fun(domain=f.domain, mapping=f.mapping, onefun=new_fun)

@implements(np.diff)
def diff(f, k=1, dim=0, *args, **kwargs):
    rescaleFactor = (0.5 * np.diff(f.domain))**k

    if dim == 0:
        df = f.onefun.diff(k=k, axis=dim) / rescaleFactor
    else:  # column wise
        return NotImplemented

    return Fun(domain=f.domain, mapping=f.mapping, onefun=df)

@implements(np.sum)
def sum(f, dim=0, *args, **kwargs):
    """ Definite integral of a Fun on its interval [a, b] """
    rescaleFactor = 0.5 * np.diff(f.domain)
    return f.onefun.sum(dim=dim, *args, **kwargs) * rescaleFactor

@implements(np.cumsum)
def cumsum(f, m=1, *args, **kwargs):
    """ Definite integral of a Fun on its interval [a, b] """
    rescaleFactor = 0.5 * np.diff(f.domain)
    nf = f.onefun.cumsum(m=m, *args, **kwargs) * rescaleFactor
    return Fun(domain=f.domain, mapping=f.mapping, onefun=nf)

@implements(np.inner)
def inner(f, g):
    rescaleFactor = 0.5 * np.diff(f.domain)
    # TODO: check that both f and g are of the same underlying type!
    # return innerproduct(f.onefun, g.onefun) * rescaleFactor
    return f.onefun.innerproduct(g.onefun) * rescaleFactor

@implements(np.roll)
def roll(f, a):
    if not f.istrig:
        return NotImplemented
    nf = circshift(f.onefun, a)
    return Fun(domain=f.domain, mapping=f.mapping, onefun=nf)

def roots(f, *args, **kwargs):
    r = f.onefun.roots(*args, **kwargs)
    return f.mapping(r)

def minandmax(f, *args, **kwargs):
    vals, pos = f.onefun.minandmax(*args, **kwargs)
    return vals, f.mapping(pos)

def plotcoeffs_trig(f, loglog=False):
    """ Return data to plot the coefficients of a trigtech """
    ac = np.abs(f.coeffs)
    n, m = ac.shape
    isEven = not (n & 1)
    if isEven:
        coeffIndex = np.arange(-n//2, n//2)
    else:
        coeffIndex = np.arange(-(n-1)//2, (n-1)//2+1)

    if f.get_vscale() == 0:
        ac += f.eps

    if not loglog:
        # normalized wave numbers
        normalizedWaveNumber = coeffIndex * (2. * np.pi) / np.diff(f.domain)
    else:
        if isEven:
            cPos = ac[n//2:, :]
            cNeg = ac[n//2::-1, :]
        else:
            cPos = ac[(n+1)//2-1:, :]
            cNeg = ac[(n+1)//2-1::-1, :]

        coeffIndexPos = np.arange(cPos.size-1)
        coeffIndexNeg = np.arange(cNeg.size-1)
        waveNumber = np.hstack((coeffIndexPos, np.nan, coeffIndexNeg))
        normalizedWaveNumber = waveNumber*(2*np.pi)/np.diff(f.domain)

    return normalizedWaveNumber, ac

def plotcoeffs(f, loglog=False):
    if f.istrig:
        return plotcoeffs_trig(f, loglog=loglog)
    else:
        assert False,''
