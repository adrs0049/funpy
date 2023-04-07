#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
from copy import deepcopy
from numbers import Number

from fun import Fun
from fun import h1norm, norm, norm2, sturm_norm, sturm_norm_alt
from states.base_state import BaseState
from states.parameter import Parameter
from support.cached_property import lazy_property
from states.namespace import Namespace


class TwoParameterState(BaseState):
    __short_name__ = 'TwoParState'

    def __init__(self, p1=None, p2=None, u=None, phi=None, *args, **kwargs):

        # Signature -> (y, mu) with y = (x, phi, lambda) -> (x, phi, lambda, mu)
        signature = [BaseState.SpaceType.REAL,   # a = lambda
                     BaseState.SpaceType.REAL,   # b = mu
                     BaseState.SpaceType.CHEB,   # x
                     BaseState.SpaceType.CHEB]   # phi

        super().__init__(signature=signature, *args, **kwargs)

        # Setup functions - TODO implement np.equal for state!
        if p1 is not None and p2 is not None and u is not None:
            self.funcs[0] = deepcopy(u)
            self.funcs[1] = np.zeros_like(u)
            self.reals[0] = deepcopy(p1)
            self.reals[1] = deepcopy(p2)

        if phi is not None:
            self.funcs[1] = deepcopy(phi)

            # make sure both functions have the same size!
            N = max(self.funcs[0].n, self.funcs[1].n)
            self.funcs[0].prolong(N)
            self.funcs[1].prolong(N)

        if self.reals[0].name == self.reals[1].name:
            raise RuntimeError(f"Parameters {self.reals[0].name} and {self.reals[1].name} must be different!")

        # Sync namespace
        self.sync_ns()

    @classmethod
    def from_coeffs(cls, coeffs, pname1, pname2, n_funs, *args, **kwargs):
        assert (coeffs.size - 2) % n_funs == 0, 'Coeffs does not have correct size!'
        soln = Fun.from_coeffs(coeffs[:-2], n_funs, *args, **kwargs)
        ap1 = Parameter(**{pname1: np.real(coeffs[-2].item())})
        # The last parameter is the current continuation variable.
        ap2 = Parameter(**{pname2: np.real(coeffs[-1].item())})
        return cls(p1=ap1, p2=ap2, u=np.real(soln))

    @classmethod
    def from_fun(cls, fun, pname1, pvalue1, pname2, pvalue2, *args, **kwargs):
        ap1 = Parameter(**{pname1: float(pvalue1)})
        ap2 = Parameter(**{pname2: float(pvalue2)})
        return cls(p1=ap1, p2=ap2, u=np.real(fun))

    @classmethod
    def from_state(cls, other, pname1, pvalue1, pname2, pvalue2, *args, **kwargs):
        ap1 = Parameter(**{pname1: float(pvalue1)})
        ap2 = Parameter(**{pname2: float(pvalue2)})
        pars = {n: v for n, v in other.items()}
        return cls(p1=ap1, p2=ap2, u=other.u, *args, **pars, **kwargs)

    def __len__(self):
        # TODO?
        return self.u.n

    def flatten(self):
        return np.hstack((self.funcs[0].flatten(), self.reals[0].value, self.reals[1].value))

    @lazy_property
    def rank(self):
        return np.product(self.funcs[0].shape) + np.product(self.funcs[1].shape) + 1

    @property
    def domain(self):
        return self.funcs[0].domain

    @property
    def shape(self):
        return (self.funcs[0].shape[0] + self.funcs[1].shape[0], self.funcs[0].shape[1])

    @property
    def u(self):
        return self.funcs[0]

    @property
    def phi(self):
        return self.funcs[1]

    @property
    def a(self):
        return self.reals[1].value

    @property
    def b(self):
        return self.reals[0].value

    @property
    def bname(self):
        return self.reals[0].name

    @property
    def par(self):
        return self.reals[1]

    @property
    def cpar(self):
        return self.reals[1].name

    def plot_values(self, cidx=0, norm_type='Sturm', p=2, k=1):
        if norm_type == 'Sturm':
            # Computes f'(a) || f ||
            return np.real(self.b), sturm_norm(self.u[cidx], p=p)
        if norm_type == 'Sturm0':
            # Computes f'(a) || f ||
            u0 = self.u - np.sum(np.sum(self.u)) / self.u.shape[1]
            return np.real(self.b), sturm_norm(u0[cidx], p=p)
        elif norm_type == 'SturmNeumann':
            return np.real(self.b), sturm_norm_alt(self.u[cidx], p=p)
        elif norm_type == 'SturmNeumann0':
            # remove the combined mass from the solution state
            u0 = self.u[cidx] - (1. / np.diff(self.u.domain)).item() * np.sum(np.sum(self.u[cidx]))
            return np.real(self.b), sturm_norm_alt(u0, p=p)
        elif norm_type == 'Wkp':
            return np.real(self.b), h1norm(self.u[cidx], p=p, k=k)
        else:
            return np.real(self.b), norm(self.u[cidx], p=p)

    """ Internal interface """
    def __repr__(self):
        masses = self.mass()
        return '%s[%s* = %.4g, %s = %.4g; n = %d; |u| = %.3g; |φ| = %.3g; ∫u = %.3g; ∫φ = %.3g; ε = %.2g]' % \
                (type(self).__short_name__, self.cpar, np.real(self.reals[1].value),
                 self.reals[0].name, np.real(self.reals[0].value),
                 self.u.shape[0], norm(self.u),
                 norm(self.phi), masses[0], masses[1], max(self.funcs[0].eps, self.funcs[1].eps))

    def __str__(self):
        return self.__repr__()
