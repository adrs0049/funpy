#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
from enum import IntEnum
from copy import deepcopy
from numbers import Number

import numpy as np

from ..fun import norm2, minandmax, innerw
from .namespace import Namespace
from .parameter import Parameter


HANDLED_FUNCTIONS = {}


class BaseState(np.lib.mixins.NDArrayOperatorsMixin):
    """
    Implements Cartesian products of inner product spaces, of finite and
    infinite dimension.

    Example:
        1) R x X
        2) R x R x X x Y

    The parameters defined in R must be taken from the parameters
    defined in the namespace.

    """
    class SpaceType(IntEnum):
        REAL = 1
        CHEB = 2
        TRIG = 3
        UNKW = 99

        @classmethod
        def from_value(cls, value):
            try:
                return {item.value: item for item in cls}[value]
            except KeyError:
                return cls.UNKW

    def __init__(self, signature=[], *args, **kwargs):
        # Space signature
        self.signature = signature

        # Weights
        self.weights = kwargs.pop('weights', np.ones(len(self.signature)))

        # Create functions
        self.funcs = kwargs.pop('funcs', np.empty(self.number_func, dtype=object))
        self.reals = kwargs.pop('reals', np.empty(self.number_real, dtype=object))

        # Now only the functions and constants should remain in the kwargs
        self.ns = kwargs.pop('ns', None)
        if self.ns is None:  # If namespace is None construct one!
            self.ns = Namespace()
            # load everything remaining in kwargs into the state's namespace
            for par, par_value in kwargs.items():
                self.ns[par] = Parameter(**{par: par_value})

    def sync_ns(self):
        # Remove reals[0] name from ns
        try:
            # Delete the names of the parameters from the namespace!
            for r in self.reals:
                del self.ns[r.name]
        except KeyError:
            pass

    @property
    def number_real(self):
        return self.signature.count(BaseState.SpaceType.REAL)

    @property
    def number_cheb(self):
        return self.signature.count(BaseState.SpaceType.CHEB)

    @property
    def number_trig(self):
        return self.signature.count(BaseState.SpaceType.TRIG)

    @property
    def number_func(self):
        return self.signature.count(BaseState.SpaceType.CHEB) + \
                self.signature.count(BaseState.SpaceType.TRIG)

    @property
    def columns(self):
        return np.sum([func.m for func in self.funcs])

    @property
    def rank(self):
        return np.product(self.funcs[0].shape)

    @property
    def size(self):
        return np.product(self.u.shape) + 1

    @property
    def is_constant(self):
        return self.u.n == 1

    @property
    def istrig(self):
        return self.funcs[0].istrig

    @property
    def fshape(self):
        return len(self.funcs), len(self.reals),

    @property
    def shape(self):
        return self.funcs[0].shape

    @property
    def bname(self):
        return ''

    @property
    def pars(self):
        rstr = ''
        for p_name, p_value in self.items():
            rstr += '{0:s} = {1:.4g} '.format(p_name, p_value)
        return rstr

    def mass(self):
        return [np.sum(np.sum(f)) for f in self.funcs]

    def flip(self):
        self *= -1.0
        return self

    def norm(self, p=2, norm_function=norm2, **kwargs):
        norm = 0.0
        for i, function in enumerate(self.funcs):
            norm += self.weights[i] * norm_function(function, p=p, **kwargs)

        for i, real in enumerate(self.reals, start=len(self.funcs)):
            norm += self.weights[i] * float(real**p)

        return np.power(norm, 1. / p)

    def normalize(self, p=2, **kwargs):
        this_norm = self.norm(p=p, **kwargs)
        if this_norm > 1e-14:
            for function in self.funcs:
                function /= this_norm
            for real in self.reals:
                real /= this_norm

        return self

    def variation(self):
        """ TODO use proper definition of variation for smooth functions """
        if self.u.shape[0] > 1:
            vals, pos = minandmax(self.u)
            return np.real(np.sum(np.diff(vals, axis=0)))
        return 0.0

    def happy(self, *args, **kwargs):
        ishappy = True
        cutoff = 0
        for func in self.funcs:
            _ishappy, ncutoff = func.happy(*args, **kwargs)
            cutoff = max(ncutoff, cutoff)
            ishappy &= _ishappy
        return ishappy, cutoff

    def prolong(self, n):
        for i, function in enumerate(self.funcs):
            self.funcs[i] = function.prolong(n)
        return self

    def simplify(self, *args, **kwargs):
        for i, function in enumerate(self.funcs):
            self.funcs[i] = function.simplify(*args, **kwargs)
        return self

    """ Dictionary access to the underlying namespace """
    def keys(self):
        return self.ns.keys()

    def items(self):
        for key, par in self.ns.items():
            yield (key, float(par))

        for par in self.reals:
            yield (par.name, par.value)

    def update_items(self):
        for par in self.reals:
            yield (par.name, par.value)

    """ I/O support """
    def writeHDF5(self, fh):
        fh.create_dataset('weights', data=self.weights)
        fh.create_dataset('signature', data=np.asarray(self.signature, dtype=int))

        for i, func in enumerate(self.funcs):
            grp = fh.create_group('funcs{}'.format(i))
            func.writeHDF5(grp)

        for i, real in enumerate(self.reals):
            grp = fh.create_group('reals{}'.format(i))
            real.writeHDF5(grp)

        # Write the namespace
        grp = fh.create_group('ns')
        self.ns.writeHDF5(grp)

    """ Internal interface """
    def __array__(self):
        # State is (x, phi, a, b)
        return np.hstack([f.coeffs.flatten(order='F') for f in self.funcs] + \
                         [np.asarray(r) for r in self.reals])

    def __repr__(self):
        return 'BaseState = {}'.format(self.funcs[0])

    def __str__(self):
        return 'BaseState = {}'.format(self.funcs[0])

    def __getitem__(self, idx):
        m = self.u.m
        return self.funcs[idx // m][idx % m]

    def __iter__(self):
        self.ipos = 0
        return self

    def __next__(self):
        if self.ipos >= self.columns:
            raise StopIteration
        self.ipos += 1
        return self[self.ipos-1]

    def __setitem__(self, key, value):
        # TODO: this is not elegant at all; and setitem and getitem do
        # different things now. FIXME
        self.ns[key] = value

    def __deepcopy__(self, memo):
        id_self = id(self)
        _copy = memo.get(id_self)
        if _copy is None:
            _copy = type(self)(
                funcs=deepcopy(self.funcs, memo),
                reals=deepcopy(self.reals, memo),
                weights=self.weights,
                ns=deepcopy(self.ns, memo))
            memo[id_self] = _copy
        return _copy

    def __getattr__(self, name):
        """ Delegate lookup of some symbols to the underlying function objects """
        if not all([hasattr(f, name) for f in self.funcs]):
            raise AttributeError(name)
        attr = [getattr(f, name) for f in self.funcs]
        # If we have a single element -> return the element directly!
        if len(attr) == 1: return attr[0]
        else: return attr

    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        # TODO: why is this failing for np.dot?
        #if not all(issubclass(t, self.__class__) for t in types):
        #    return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def __array_ufunc__(self, numpy_ufunc, method, *inputs, **kwargs):
        out = kwargs.get('out', ())
        for x in inputs + out:
            if not isinstance(x, (Number, np.ndarray, type(self))):
                return NotImplemented

        if method == '__call__':
            # Get the underlying sizes
            nr = self.reals.size
            nf = self.funcs.size

            if out:
                for i in range(nr):
                    tlist = [x.reals[i] if isinstance(x, type(self)) else x for x in inputs]
                    numpy_ufunc(*tlist, out=out[0].reals[i])

                for i in range(nf):
                    tlist = [x.funcs[i] if isinstance(x, type(self)) else x for x in inputs]
                    numpy_ufunc(*tlist, out=out[0].funcs[i])

                # Update the namespace
                out[0].sync_ns()
                return out[0]
            else:
                nreals = np.empty(nr, dtype=object)
                nfuncs = np.empty(nf, dtype=object)

                # Compute the new real and function data
                for i in range(nr):
                    tlist = [x.reals[i] if isinstance(x, type(self)) else x for x in inputs]
                    nreals[i] = numpy_ufunc(*tlist)

                for i in range(nf):
                    tlist = [x.funcs[i] if isinstance(x, type(self)) else x for x in inputs]
                    nfuncs[i] = numpy_ufunc(*tlist)

                return type(self)(funcs=nfuncs, reals=nreals, ns=self.ns)
        else:
            return NotImplemented


def implements(np_function):
    """ Register an __array_function__ implementation """
    def decorator(func):
        HANDLED_FUNCTIONS[np_function] = func
        return func
    return decorator


@implements(np.zeros_like)
def zeros_like(state):
    funcs = np.empty_like(state.funcs)
    reals = np.empty_like(state.reals)

    for i, func in enumerate(state.funcs):
        funcs[i] = np.zeros_like(func)

    for i, real in enumerate(state.reals):
        reals[i] = np.zeros_like(real)

    return type(state)(funcs=funcs, reals=reals)


@implements(np.ones_like)
def ones_like(state):
    funcs = np.empty_like(state.funcs)
    reals = np.empty_like(state.reals)

    for i, func in enumerate(state.funcs):
        funcs[i] = np.ones_like(func)

    for i, real in enumerate(state.reals):
        reals[i] = np.ones_like(real)

    return type(state)(funcs=funcs, reals=reals)


@implements(np.dot)
def dot(state1, state2):
    """ state is a Cartesian space of X x R,
        thus the inner product (state1, state2) = Int(u1, u2) + (a1, a2)
    """
    inner = 0.0
    n, m = state1.fshape
    for i in range(n):
        inner += state1.weights[i] * np.dot(state1.funcs[i], state2.funcs[i])

    for i in range(m):
        inner += state1.weights[i+n] * np.dot(state1.reals[i], state2.reals[i])

    return inner


@implements(np.inner)
def inner(state1, state2):
    """ state is a Cartesian space of X x R,
        thus the inner product (state1, state2) = Int(u1, u2) + (a1, a2)
    """
    inner = 0.0
    n, m = state1.fshape
    for i in range(n):
        ip = state1.weights[i] * np.inner(state1.funcs[i], state2.funcs[i])
        inner += np.sum(ip.diagonal()) if ip.ndim == 2 else np.sum(ip)

    for i in range(m):
        inner += state1.weights[i+n] * np.inner(state1.reals[i], state2.reals[i])

    return inner


def innerw(state1, state2):
    """ state is a Cartesian space of X x R,
        thus the inner product (state1, state2) = Int(u1, u2) + (a1, a2)
    """
    inner = 0.0
    n, m = state1.fshape
    for i in range(n):
        ip = state1.weights[i] * innerw(state1.funcs[i], state2.funcs[i])
        inner += np.sum(ip)

    for i in range(m):
        inner += state1.weights[i+n] * np.inner(state1.reals[i], state2.reals[i])

    return inner


@implements(np.real)
def real(state):
    n, m = state.fshape
    nreals = np.empty_like(state.reals)
    nfuncs = np.empty_like(state.funcs)

    for i in range(n):
        nreals[i] = np.real(state.reals[i])

    for i in range(m):
        nfuncs[i] = np.real(state.funcs[i])

    return type(state)(reals=nreals, funcs=nfuncs, ns=state.ns)


@implements(np.imag)
def imag(state):
    n, m = state.fshape
    nreals = np.empty_like(state.reals)
    nfuncs = np.empty_like(state.funcs)

    for i in range(n):
        nreals[i] = np.imag(state.reals[i])

    for i in range(m):
        nfuncs[i] = np.imag(state.funcs[i])

    return type(state)(reals=nreals, funcs=nfuncs, ns=state.ns)


@implements(np.conj)
def conj(state):
    n, m = state.fshape
    nreals = np.empty_like(state.reals)
    nfuncs = np.empty_like(state.funcs)

    for i in range(n):
        nreals[i] = np.conj(state.reals[i])

    for i in range(m):
        nfuncs[i] = np.conj(state.funcs[i])

    return type(state)(reals=nreals, funcs=nfuncs, ns=state.ns)


@implements(np.sum)
def sum(state):
    return np.sum(state.u)
