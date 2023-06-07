#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen

from numbers import Number
from copy import deepcopy
import numpy as np

# Local imports
from .cheb import chebtech
from .cheb import innerw as cheb_innerw
from .mapping import Mapping
from .trig import trigtech
from .trig import innerw as trig_innerw
from .functional import Functional
from .trig.operations import circconv, circshift, trig_adhesion

from .trig import trigtech
from .ultra import ultra2ultra

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

        # Make sure onefun is a function object!
        assert isinstance(self.onefun, chebtech) or isinstance(self.onefun, trigtech), ''

    def __construct(self, *args, **kwargs):
        fun_type = kwargs.pop('type', 'cheb')
        if fun_type == 'trig':
            self.onefun = trigtech(*args, **kwargs)
        else:  # we will simply default to cheb
            self.onefun = chebtech(*args, **kwargs)

    @property
    def istrig(self):
        return self.onefun.istrig

    @property
    def type(self):
        return 'trig' if self.istrig else 'cheb'

    @property
    def ndim(self):
        return self.onefun.ndim

    """ Return the points at which the chebtech is sampled at """
    @property
    def x(self):
        return self.mapping.fwd(self.onefun.x)

    @property
    def hscale(self):
        return np.linalg.norm(self.domain, np.inf)

    def __eq__(self, other):
        return np.all(self.domain == other.domain) and np.all(self.onefun == other.onefun) \
                and np.all(self.mapping == other.mapping)

    def simplify(self, *args, **kwargs):
        self.onefun = self.onefun.simplify(*args, **kwargs)
        return self

    def prolong(self, Nout):
        self.onefun = self.onefun.prolong(Nout)
        return self

    def norm(self, p=2, **kwargs):
        return norm(self, p=p, **kwargs)

    def seteps(self, value):
        self.onefun.eps = value

    def flatten(self):
        return self.coeffs.flatten(order='F')

    def __len__(self):
        return len(self.onefun)

    def __str__(self):
        return '%sfun column (%d pieces) on %s at %s points.' % (self.type, self.m, self.domain, len(self))

    def __repr__(self):
        with np.printoptions(precision=16):
            return f"{self.__class__.__name__}(op={repr(self.onefun.values)}, domain={repr(self.domain)}, type={repr(self.type)})"

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
        if not all(issubclass(t, self.__class__) or isinstance(t, Functional) for t in types):
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
    def writeHDF5(self, fh):
        self.onefun.writeHDF5(fh)
        fh.attrs['domain'] = self.domain
        fh.attrs['type'] = type(self.onefun).__name__

    @classmethod
    def from_hdf5(cls, hdf5_file):
        domain = hdf5_file.attrs['domain']
        ftype = hdf5_file.attrs['type']
        fun = eval('{}.from_hdf5(hdf5_file[\'poly\'])'.format(ftype))
        return cls(onefun=fun, domain=domain)

    @classmethod
    def from_coeffs(cls, coeffs, n_funs, type='cheb', order='F', *args, **kwargs):
        assert coeffs.size % n_funs == 0, 'Array must fit!'
        m = coeffs.size // n_funs
        return cls(coeffs=coeffs.reshape((m, n_funs), order=order),
                   simplify=False, eps=np.finfo(float).eps,
                   type=type, *args, **kwargs)

    @classmethod
    def from_values(cls, values, n_funs, type='cheb', *args, **kwargs):
        assert values.size % n_funs == 0, 'Array must fit!'

        if type == 'cheb':
            nf = chebtech.from_values(values, simplify=False)
        elif type == 'trig':
            nf = trigtech.from_values(values, simplify=False)
        else:
            raise RuntimeError("Unknown function type {0:s}!".format(type))

        return cls(onefun=nf, simplify=False, eps=np.finfo(float).eps,
                   type=type, *args, **kwargs)

    @classmethod
    def from_ultra(cls, coeffs, n_funs, lam_in, type='cheb', order='F', *args, **kwargs):
        assert coeffs.size % n_funs == 0, 'Array must fit!'
        m = coeffs.size // n_funs
        coeffs = np.asfortranarray(ultra2ultra(coeffs.reshape((m, n_funs), order=order), lam_in, 0))
        return cls(coeffs=coeffs, simplify=False, eps=np.finfo(float).eps,
                   type=type, *args, **kwargs)


def implements(np_function):
    """ Register an __array_function__ implementation """
    def decorator(func):
        HANDLED_FUNCTIONS[np_function] = func
        return func
    return decorator


@implements(np.argmax)
def argmax(f):
    return np.argmax(f.onefun)


@implements(np.zeros_like)
def zeros_like(f):
    return Fun(domain=f.domain, mapping=f.mapping, type=f.type,
               op=np.zeros((f.n, f.m), order='F'), simplify=False)


@implements(np.ones_like)
def ones_like(f):
    return Fun(domain=f.domain, mapping=f.mapping, type=f.type,
               op=np.ones((f.n, f.m), order='F'), simplify=False)


@implements(np.real)
def real(f):
    return Fun(domain=f.domain, mapping=f.mapping, onefun=np.real(f.onefun))


@implements(np.imag)
def imag(f):
    return Fun(domain=f.domain, mapping=f.mapping, onefun=np.imag(f.onefun))


def zeros(n, domain=[-1, 1], *args, **kwargs):
    """ Creates n functions equally zero. """
    return Fun(coeffs=np.zeros((1, n), dtype=float, order='F'), domain=domain, simplify=False, *args, **kwargs)


def ones(n, domain=[-1, 1], *args, **kwargs):
    """ Creates n functions equally zero. """
    return Fun(coeffs=np.ones((1, n), dtype=float, order='F'), domain=domain, simplify=False, *args, **kwargs)


def random(n, m, domain=[-1, 1], scale=1.0, *args, **kwargs):
    """ Creates n functions equally zero. """
    coeffs = scale * np.asfortranarray(1.0 - 2.0 * np.random.rand(n, m))
    return Fun(coeffs=coeffs, domain=domain, simplify=False, *args, **kwargs)


def random_decay(n, m, domain=[-1, 1], scale=1.0, *args, **kwargs):
    """ Creates n functions equally zero. """
    nn = (1 + np.arange(n))**2
    coeffs = scale * (1.0 - 2.0 * np.asfortranarray(np.random.rand(n, m) / nn[:, None]))
    return Fun(coeffs=coeffs, domain=domain, simplify=False, *args, **kwargs)


def asfun(obj, *args, **kwargs):
    if isinstance(obj, Fun):
        ftype = kwargs.pop('type', obj.type)

        if obj.type == ftype:
            return obj

        return Fun(op=lambda x: obj(x), type=ftype, domain=obj.domain, *args, **kwargs)

    elif isinstance(obj, Number) or obj.size == 1:
        return Fun(coeffs=np.reshape(np.asarray(obj, dtype=float), (1, 1), order='F'), *args, **kwargs)

    assert False, 'Unsupported type %s!' % type(obj)


@implements(np.dot)
def dot(input1, input2):
    if not np.all(input1.domain == input2.domain):
        raise ValueError("Domain mismatch %s != %s!" % (input1.domain, input2.domain))

    if isinstance(input1, Functional) and isinstance(input2, Fun):
        return np.dot(input1.coeffs, input2.coeffs)
    elif isinstance(input1, np.ndarray):
        return np.dot(input1, input2.values)
    elif isinstance(input1, Fun) and isinstance(input2, Fun):
        rescaleFactor = 0.5 * np.diff(input1.domain).item()
        return np.dot(input1.onefun, input2.onefun) * rescaleFactor
    else:
        assert False, 'Don\'t know what to do about this!'


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


def norm(f, p=2, weighted=False):
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
            return np.abs(np.sum(np.sum(f**p)))

    def detail_weighted(f, p):
        # We must have that p == 2
        assert p == 2, 'A weighted norm can only be used when p == 2!'
        return np.sum(innerw(f, f))

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
        norm_p = detail(f, p) if not weighted else detail_weighted(f, p)
        if p == 1 or p == np.inf:
            return norm_p.squeeze()
        else:
            return np.power(norm_p, 1. / p).squeeze()


def norm2(f, p=2, weighted=False):
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

    def detail_weighted(f, p):
        # We must have that p == 2
        assert p == 2, 'A weighted norm can only be used when p == 2!'
        return np.sum(innerw(f, f))

    if f.istrig:
        fR = np.real(f)
        fI = np.imag(f)
        return detail(fR, p) + detail(fI, p)
    else:
        return detail(f, p) if not weighted else detail_weighted(f, p)


def wkpnorm(f, p=2, k=1, **kwargs):
    """ Compute the k, p Sobolev norm of f """
    if k < 0:
        raise ValueError("The derivative norm index must be positive!")

    norm = norm2(f, p=p, **kwargs)
    for i in range(1, k+1):
        f = np.diff(f)
        norm += norm2(f, p=p, **kwargs)

    return np.power(norm, 1./p)


def h1norm(f, p=2, **kwargs):
    return wkpnorm(f, p=p, k=1, **kwargs)


def h2norm(f, p=2, **kwargs):
    return wkpnorm(f, p=p, k=2, **kwargs)


def normh(f):
    return np.sqrt(np.dot(f, f))


def sturm_norm(f, p=2, **kwargs):
    """ Computes the p "Sturm" norm of the function f(x) defined on the interval [a, b]

            sign[f'(a)] * || f ||_p
    """
    df = np.diff(f)
    df_norm = norm(df, p=p, **kwargs)
    snorm = np.real(np.sign(df(df.domain[0]))) * norm(f, p=p) if df_norm > 1e-7 else norm(f, p=p)
    return snorm


def sturm_norm_alt(f, p=2, **kwargs):
    """ Computes the p "Sturm" norm of the function f(x) defined on the interval [a, b]

            sign[f(a)] * || f ||_p
    """
    df = np.diff(f)
    df_norm = norm(df, p=p, **kwargs)
    snorm = np.real(np.sign(f(f.domain[0]))) * norm(f, p=p) if df_norm > 1e-7 else norm(f, p=p)
    return snorm


def normalize(u, *args, **kwargs):
    return u / norm(u, *args, **kwargs)


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
    return Fun(domain=f.domain, mapping=f.mapping, onefun=np.conj(f.onefun))


@implements(np.diff)
def diff(f, n=1, axis=0, *args, **kwargs):
    rescaleFactor = (0.5 * np.diff(f.domain))**n

    if axis == 0:
        df = np.diff(f.onefun, n=n, axis=axis) / rescaleFactor
    else:  # column wise
        return NotImplemented

    return Fun(domain=f.domain, mapping=f.mapping, onefun=df)


@implements(np.sum)
def sum(f, axis=0, *args, **kwargs):
    """ Definite integral of a Fun on its interval [a, b] """
    rescaleFactor = 0.5 * np.diff(f.domain)
    return np.sum(f.onefun, axis=axis, *args, **kwargs) * rescaleFactor


@implements(np.cumsum)
def cumsum(f, *args, **kwargs):
    """ Definite integral of a Fun on its interval [a, b] """
    rescaleFactor = 0.5 * np.diff(f.domain)
    nf = np.cumsum(f.onefun, *args, **kwargs) * rescaleFactor
    return Fun(domain=f.domain, mapping=f.mapping, onefun=nf)


@implements(np.inner)
def inner(f, g):
    if not np.all(f.domain == g.domain):
        raise ValueError("Domain mismatch %s != %s!" % (f.domain, g.domain))

    rescaleFactor = 0.5 * np.diff(f.domain)
    return np.inner(f.onefun, g.onefun) * rescaleFactor


def innerw(f, g):
    if not np.all(f.domain == g.domain):
        raise ValueError("Domain mismatch %s != %s!" % (f.domain, g.domain))

    rescaleFactor = 0.5 * np.diff(f.domain)
    if f.istrig and g.istrig:
        return trig_innerw(f.onefun, g.onefun) * rescaleFactor
    else:
        return cheb_innerw(f.onefun, g.onefun) * rescaleFactor


@implements(np.roll)
def roll(f, a):
    if not f.istrig:
        return NotImplemented
    nf = circshift(f.onefun, a)
    return Fun(domain=f.domain, mapping=f.mapping, onefun=nf)


@implements(np.hstack)
def hstack(funs):
    nf = np.hstack([f.onefun for f in funs])
    return Fun(domain=funs[0].domain, mapping=funs[0].mapping, onefun=nf)


@implements(np.copy)
def copy(fun):
    nf = np.copy(fun.onefun)
    return Fun(domain=fun.domain, mapping=fun.mapping, onefun=nf)


# @implements(np.hsplit)
# def hsplit(fun):
#     nfs = np.hsplit(fun.onefun)
#     return [Fun(domain=fun.domain, mapping=fun.mapping, onefun=nf) for nf in nfs]


def roots(f, *args, **kwargs):
    r = f.onefun.roots(*args, **kwargs)
    return f.mapping(r)


def minandmax(f, *args, **kwargs):
    vals, pos = f.onefun.minandmax(*args, **kwargs)
    return vals, f.mapping(pos)


def prolong(f, Nout):
    nf = np.copy(f)
    return nf.prolong(Nout)


def plot(f, npts=1000, *args, **kwargs):
    import matplotlib.pyplot as plt
    xs = np.linspace(*f.domain, npts)
    fig, ax = plt.subplots(*args, **kwargs)
    ax.plot(xs, f(xs))
    return fig


def plot_values(f, *args, **kwargs):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(*args, **kwargs)
    n, m = f.shape
    for i in range(m):
        ax.scatter(f.x, f.values[:, i])
    return fig


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


def plotcoeffs_cheb(f, loglog=False):
    absCoeffs = np.abs(f.coeffs)
    n, m = absCoeffs.shape
    xx = np.arange(0, n, 1)
    return xx, absCoeffs


def plotcoeffs(f, loglog=False, *args, **kwargs):
    import matplotlib.pyplot as plt
    if f.istrig:
        k, c = plotcoeffs_trig(f, loglog=loglog)
    else:
        k, c = plotcoeffs_cheb(f, loglog=loglog)

    fig, ax = plt.subplots(*args, **kwargs)
    ax.scatter(k, c)
    return fig
