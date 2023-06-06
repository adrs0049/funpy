#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
from numbers import Number
import numpy as np


class QuasiOpBlock:
    """ Represents a linear operator acting on a single function """
    def __init__(self, block, *args, **kwargs):
        self.block = block
        self._info = kwargs.pop('info')
        self._nl_info = kwargs.pop('nl_info')

        # Storage for current coefficients
        self.coeffs    = kwargs.pop('coeffs', np.empty(1, dtype=object))
        self.nl_coeffs = kwargs.pop('nl_coeffs', np.empty(1, dtype=object))
        self.icoeffs  = kwargs.pop('icoeffs', None)
        self.integrand  = kwargs.pop('integrand', None)

    @property
    def has_nonlocal(self):
        return self.icoeffs is not None

    @property
    def positive(self):
        return self.info()

    @property
    def diff_order(self):
        return self.coeffs.size

    @property
    def numNonLocal(self):
        return self.icoeffs.size if self.icoeffs is not None else 0

    def info(self):
        return self._info

    def nl_info(self):
        return self._nl_info

    def getCoeffs(self):
        """ Returns the coefficients of the derivative terms of the linear operator """
        return self.coeffs

    def getNlCoeffs(self):
        """ Returns the coefficients of the derivative terms of the linear operator """
        return self.nl_coeffs

    def getICoeffs(self):
        """ Returns the coefficients of the derivative terms of the linear operator """
        return self.icoeffs

    def getIntegrand(self):
        """ Returns the coefficients of the derivative terms of the linear operator """
        return self.integrand

    """ Allowed mathematical operations, in the vector space of operators! """
    def __pos__(self):
        return self

    def __neg__(self):
        # add current coefficients
        new_coeffs = np.empty_like(self.coeffs, dtype=object)
        for i in range(self.diff_order):
            new_coeffs[i] = -self.coeffs[i]

        # aggregation non-local coefficients
        if np.any(self._nl_info):
            nl_new_coeffs = np.empty_like(self.nl_coeffs, dtype=object)
            for i in range(nl_new_coeffs.shape[0]):
                nl_new_coeffs[i] = -self.nl_coeffs[i]
        else:
            nl_new_coeffs = np.empty(1, dtype=object)

        # add integral operators
        new_icoeffs = None
        new_integrand = None

        if self.has_nonlocal:
            new_icoeffs = np.empty_like(self.icoeffs, dtype=object)
            for i in range(self.icoeffs.size):
                new_icoeffs = -self.icoeffs[i]

            new_integrand = np.empty_like(self.integrand, dtype=object)
            for i in range(self.integrand.size):
                new_integrand = -self.integrand[i]

        return QuasiOpBlock(self.block, coeffs=new_coeffs, icoeffs=new_icoeffs,
                            integrand=new_integrand, nl_coeffs=nl_new_coeffs,
                            info=self.info(), nl_info=self.nl_info())

    def __add__(self, other):
        # add current coefficients
        new_coeffs = np.empty_like(self.coeffs, dtype=object)
        for i in range(self.diff_order):
            new_coeffs[i] = self.coeffs[i] + other.coeffs[i]

        # aggregation non-local coefficients -> TODO IMPROVE ME!!!
        if np.any(self._nl_info) or np.any(other._nl_info):
            nl_new_coeffs = np.empty_like(self.nl_coeffs, dtype=object)

            if np.any(self._nl_info) and np.any(other._nl_info):
                for i in range(nl_new_coeffs.shape[0]):
                    nl_new_coeffs[i] = self.nl_coeffs[i] + other.nl_coeffs[i]
            elif np.any(self._nl_info):
                for i in range(nl_new_coeffs.shape[0]):
                    nl_new_coeffs[i] = self.nl_coeffs[i]
            elif np.any(other._nl_info):
                for i in range(nl_new_coeffs.shape[0]):
                    nl_new_coeffs[i] = other.nl_coeffs[i]
        else:
            nl_new_coeffs = np.empty(1, dtype=object)

        # add integral operators
        new_icoeffs = None
        new_integrand = None

        # If one of the operators has the non-local property set ->
        # see whether we need to add or just take the one that has one!
        # TODO: Should we just by default set the coefficient to zero?
        if self.has_nonlocal and other.has_nonlocal:
            new_icoeffs = np.empty_like(self.icoeffs, dtype=object)
            for i in range(self.icoeffs.size):
                new_icoeffs[i] = other.icoeffs[i] + self.icoeffs[i]

            new_integrand = np.empty_like(self.integrand, dtype=object)
            for i in range(self.integrand.size):
                new_integrand[i] = other.integrand[i] + self.integrand[i]

        elif self.has_nonlocal and not other.has_nonlocal:
            new_icoeffs = self.icoeffs
            new_integrand = self.integrand

        elif not self.has_nonlocal and other.has_nonlocal:
            new_icoeffs = other.icoeffs
            new_integrand = other.integrand

        new_info = self.info() | other.info()
        new_nl_info = self.nl_info() | other.nl_info()
        return QuasiOpBlock(self.block, coeffs=new_coeffs, icoeffs=new_icoeffs,
                            integrand=new_integrand, nl_coeffs=nl_new_coeffs,
                            info=new_info, nl_info=new_nl_info)

    def __sub__(self, other):
        return self.__add__(-other)

    def __iadd__(self, other):
        raise RuntimeError("+= is not allowed for quasiBlock!")

    def __mul__(self, other):
        """ Scalar multiplication in the vector space of operators! """
        if not isinstance(other, Number):
            raise ValueError("quasiBlock.__mul__ other must be a real!")

        # add current coefficients
        new_coeffs = np.empty_like(self.coeffs, dtype=object)
        for i in range(self.diff_order):
            new_coeffs[i] = other * self.coeffs[i]

        # aggregation non-local coefficients
        if np.any(self._nl_info):
            nl_new_coeffs = np.empty_like(self.nl_coeffs, dtype=object)
            for i in range(nl_new_coeffs.shape[0]):
                nl_new_coeffs[i] = other * self.nl_coeffs[i]
        else:
            nl_new_coeffs = np.empty(1, dtype=object)

        # add integral operators
        new_icoeffs = None
        new_integrand = None
        if self.has_nonlocal:
            new_icoeffs = np.empty_like(self.icoeffs, dtype=object)
            for i in range(self.icoeffs.size):
                new_icoeffs[i] = other * self.icoeffs[i]

            new_integrand = np.empty_like(self.integrand, dtype=object)
            for i in range(self.integrand.size):
                new_integrand[i] = other * self.integrand[i]

        new_info = self.info() | other.info()
        new_nl_info = self.nl_info() | other.nl_info()
        return QuasiOpBlock(self.block, coeffs=new_coeffs, icoeffs=new_icoeffs,
                            integrand=new_integrand, nl_coeffs=nl_new_coeffs,
                            info=new_info, nl_info=new_nl_info)

    def __rmul__(self, other):
        return other * self
