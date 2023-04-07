#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
import re
from numbers import Number
from ac.support import Namespace

from fun import Fun, asfun
from colloc.quasiOpBlock import QuasiOpBlock


class LinBlock:
    """
    Represents a linear operator acting on a single function of the form:

                       d^n u
    L[u] := a^n(x, u) ------ + ... + a^0(x, u) + c(x, u) ∫ d(x, u) u(x) dx
                       dx^n

    where a^i(x) i = [0, n], are continuous functions, and c(x) and d(x) are
    continuous functions related to a possibly non-local operator.

    """
    def __init__(self, ns, block, *args, **kwargs):
        # local namespace
        self.ns = ns

        # store block
        self.block = block

        # see whether we need to print
        self.debug = kwargs.pop('debug', False)

        # Collect the functions in the namespace
        self.coeffs   = kwargs.pop('coeffs', None)
        self.info     = kwargs.pop('info', np.zeros(4).astype(bool))

        # FIXME reimplement for new version
        self.nlcoeffs = kwargs.pop('nlcoeffs', None)
        self.nl_info  = kwargs.pop('nl_info', np.zeros(2).astype(bool))

        # Integral operators as part of this block
        self.icoeffs   = kwargs.pop('icoeffs', None)
        self.integrand = kwargs.pop('integrand', None)

    @property
    def has_nonlocal(self):
        return self.icoeffs is not None

    @property
    def positive(self):
        return self.info()

    def build(self, src):
        self.coeffs = np.empty(src.diffOrder + 1, dtype=object)
        self.info = None

        # Deal with the differential operators
        for op in src.dops:
            try:
                self.coeffs[op.order] = self.ns[op.symbol_name]
            except KeyError:
                raise RuntimeError("Could not find symbol \"{0:s}\"".format(op.symbol_name))

        # Look up information
        try:
            self.info = self.ns[src.symbol_name]
        except KeyError:
            raise RuntimeError("Could not find symbol \"{0:s}\"".format(src.symbol_name))

        # Look up the non-local operators
        if src.numNonLocal > 0:
            self.icoeffs   = np.empty(src.numNonLocal, dtype=object)
            self.integrand = np.empty(src.numNonLocal, dtype=object)

            for i, op in enumerate(src.iops):
                try:
                    self.icoeffs[i] = self.ns[op.symbol_name_c]
                except KeyError:
                    raise RuntimeError("Could not find symbol \"{0:s}\"".format(op.symbol_name_c))

                try:
                    self.integrand[i] = self.ns[op.symbol_name_i]
                except KeyError:
                    raise RuntimeError("Could not find symbol \"{0:s}\"".format(op.symbol_name_i))

    def quasi(self, u, *args, **kwargs):
        """
            Returns the quasi operator of this particular linear block.
            This essentially computes the current coefficients a^i(x, u)
            and c(x, u), d(x, u).
        """
        eps = kwargs.get('eps', np.finfo(float).eps)
        new_coeffs = np.empty(len(self.coeffs), dtype=object)
        for i, coeff in enumerate(self.coeffs):
            new_coeffs[i] = coeff(*u).simplify(eps=eps)
            #print('i = ', i, ' : ', coeff, ' : ', new_coeffs[i], ' : ', asfun(new_coeffs[i], type='trig'))

        # if np.any(self.nl_info()):
        #     new_nl_coeffs = np.empty(len(self.nlcoeffs), dtype=object)
        #     for i, coeff in enumerate(self.nlcoeffs):
        #         new_nl_coeffs[i] = coeff(*u).simplify(eps=eps)
        # else:
        #     new_nl_coeffs = np.empty(1, dtype=object)

        # update the integral term
        new_icoeffs = None
        new_integrand = None

        if self.has_nonlocal:
            new_icoeffs = np.empty(len(self.icoeffs), dtype=object)
            new_integrand = np.empty(len(self.integrand), dtype=object)

            for i, coeffs in enumerate(self.icoeffs):
                new_icoeffs[i] = coeffs(*u).simplify(eps=eps)

            for i, integ in enumerate(self.integrand):
                new_integrand[i] = integ(*u).simplify(eps=eps)

        return QuasiOpBlock(self.block, info=np.asarray(self.info()),
                            nl_info=np.asarray(self.nl_info),
                            coeffs=new_coeffs, icoeffs=new_icoeffs,
                            integrand=new_integrand)

    def __str__(self):
        return 'LinBlock[%s]' % self.block

    def __repr__(self):
        return "LinBlock[%s]" % self.block
