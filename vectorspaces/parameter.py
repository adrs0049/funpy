#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
from copy import deepcopy
from numbers import Number


HANDLED_FUNCTIONS = {}


class Parameter(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, *args, **kwargs):
        self.name = None
        self.value = None

        assert len(kwargs) == 1, ''
        for key, value in kwargs.items():
            setattr(self, 'name', key)
            setattr(self, 'value', value)

    @classmethod
    def from_real(cls, name, value):
        return cls(**{name: value})

    def __deepcopy__(self, memo):
        id_self = id(self)
        _copy = memo.get(id_self)
        if _copy is None:
            _copy = type(self)(**{self.name: self.value})
            memo[id_self] = _copy
        return _copy

    def __str__(self):
        return '{0:s} = {1:.2g}'.format(self.name, self.value)

    def __repr__(self):
        return '{0:s}[{1:s} = {2:.2g}]'.format(type(self).__name__, self.name, self.value)

    def __int__(self):
        return int(self.value)

    def __float__(self):
        return float(self.value)

    def __array__(self):
        return np.asarray(self.value)

    def __array_ufunc__(self, numpy_ufunc, method, *inputs, **kwargs):
        # TODO: check that parameter names are the same!!
        out = kwargs.get('out', ())
        for x in inputs + out:
            if not isinstance(x, (Number, np.ndarray, type(self))):
                return NotImplemented

        if method == '__call__':
            ipts = [np.asarray(x) for x in inputs]
            if out:
                out[0].value = numpy_ufunc(*ipts)
                return out[0]
            else:
                return type(self)(**{self.name: numpy_ufunc(*ipts)})
        else:
            return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    @classmethod
    def from_hdf5(cls, hdf5_file):
        name = hdf5_file.attrs["name"]
        value = hdf5_file.attrs["value"]
        return cls(**{name: value})

    def writeHDF5(self, fh):
        fh.attrs["name"] = self.name
        fh.attrs["value"] = self.value


def implements(np_function):
    """ Register an __array_function__ implementation """
    def decorator(func):
        HANDLED_FUNCTIONS[np_function] = func
        return func
    return decorator


@implements(np.zeros_like)
def zeros_like(p):
    return type(p)(**{p.name: 0.0})


@implements(np.ones_like)
def ones_like(p):
    return type(p)(**{p.name: 1.0})


@implements(np.dot)
def dot(p1, p2):
    return p1.value * p2.value


@implements(np.inner)
def inner(p1, p2):
    return p1.value * p2.value


@implements(np.real)
def real(p):
    return type(p)(**{p.name: np.real(p.value)})


@implements(np.imag)
def imag(p):
    return type(p)(**{p.name: np.imag(p.value)})


@implements(np.conj)
def conj(p):
    return type(p)(**{p.name: np.conj(p.value)})


@implements(np.sum)
def sum(p):
    return type(p)(**{p.name: p.value})


@implements(np.around)
def around(p, *args, **kwargs):
    return np.around(p.value, *args, **kwargs)
