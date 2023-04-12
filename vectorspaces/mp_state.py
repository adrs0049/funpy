#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
from copy import deepcopy
from numbers import Number
import numpy as np

HANDLED_FUNCTIONS = {}


class MoorePenroseState(np.lib.mixins.NDArrayOperatorsMixin):

    def __init__(self, u, v, *args, **kwargs):
        # The main state
        self.u = u

        # The tangent
        self.v = v

    @property
    def state(self):
        return self.u

    @property
    def tangent(self):
        return self.v

    def __deepcopy__(self, memo):
        id_self = id(self)
        _copy = memo.get(id_self)
        if _copy is None:
            _copy = type(self)(u=deepcopy(self.u, memo), v=deepcopy(self.v, memo))
            memo[id_self] = _copy

        return _copy

    def __getitem__(self, idx):
        return self.u[idx]

    def __iter__(self):
        return iter(self.u)

    def __next__(self):
        return next(self.u)

    def __getattr__(self, name):
        """ Delegate lookup of some symbols to the underlying function objects """
        if not hasattr(self.u, name):
            raise AttributeError(name)
        return getattr(self.u, name)

    def __array__(self):
        return np.asarray(self.u)

    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def __array_ufunc__(self, numpy_ufunc, method, *inputs, **kwargs):
        out = kwargs.get('out', ())
        for x in inputs + out:
            if not isinstance(x, (Number, np.ndarray, type(self))):
                return NotImplemented

        if method == '__call__':
            if out:
                tlist = [x.u if isinstance(x, type(self)) else x for x in inputs]
                numpy_ufunc(*tlist, out=out[0].u)

                tlist = [x.v if isinstance(x, type(self)) else x for x in inputs]
                numpy_ufunc(*tlist, out=out[0].v)

                # Normalize the tangent
                out[0].v = out[0].v.normalize()

                return out[0]

            else:
                tlist = [x.u if isinstance(x, type(self)) else x for x in inputs]
                nu = numpy_ufunc(*tlist)

                tlist = [x.v if isinstance(x, type(self)) else x for x in inputs]
                nv = numpy_ufunc(*tlist)

                # Normalize the tangent
                nv = nv.normalize()

                return MoorePenroseState(u=nu, v=nv)

        else:
            return NotImplemented

    def __repr__(self):
        return 'MpState(u={}, v={})'.format(self.u, self.v)

    def __str__(self):
        return 'MpState(u={}, v={})'.format(self.u, self.v)

    def writeHDF5(self, fh):
        """ I/O support """

        grp = fh.create_group('u')
        self.u.writeHDF5(grp)

        grp = fh.create_group('v')
        self.v.writeHDF5(grp)


def implements(np_function):
    """ Register an __array_function__ implementation """
    def decorator(func):
        HANDLED_FUNCTIONS[np_function] = func
        return func
    return decorator


@implements(np.zeros_like)
def zeros_like(state):
    return MoorePenroseState(u=np.zeros_like(state.u), v=np.zeros_like(state.v))


@implements(np.ones_like)
def ones_like(state):
    return MoorePenroseState(u=np.ones_like(state.u), v=np.ones_like(state.v))


@implements(np.real)
def real(state):
    return MoorePenroseState(u=np.real(state.u), v=np.real(state.v))


@implements(np.imag)
def imag(state):
    return MoorePenroseState(u=np.imag(state.u), v=np.imag(state.v))


@implements(np.conj)
def conj(state):
    return MoorePenroseState(u=np.conj(state.u), v=np.conj(state.v))


@implements(np.dot)
def dot(state1, state2):
    return np.dot(state1.u, state2.u)


@implements(np.inner)
def inner(state1, state2):
    return np.inner(state1.u, state2.u)


@implements(np.sum)
def sum(state):
    return np.sum(state.u)
