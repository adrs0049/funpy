#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
import re
from numbers import Number
from ac.support import Namespace

from fun import Fun
from colloc.quasiOpBlock import QuasiOpBlock


class BiLinBlock:
    """ Represents a bilinear form acting on a single function """
    def __init__(self, ns, block, *args, **kwargs):
        # local namespace
        self.ns = ns

        # store block
        self.block = block

        # rounding for coefficient terms
        self.eps = kwargs.pop('eps', 1.48e-8)

        # TODO: why need to keep this around! ?
        self.diff_order = kwargs.pop('diff_order', 0)

        # see whether we need to print
        self.debug = kwargs.pop('debug', False)

        # Collect the functions in the namespace
        self.coeffs   = kwargs.pop('coeffs', np.empty(self.diff_order + 1, dtype=object) if self.diff_order >= 0 else [])
        self.info     = kwargs.pop('info', None)
        # Non-local term coefficients!
        self.nlcoeffs = kwargs.pop('nlcoeffs', np.empty(self.diff_order, dtype=object) if self.diff_order >= 0 else [])
        self.nl_info  = kwargs.pop('nlinfo', None)
        self.integral = kwargs.pop('integral', None)

    @property
    def has_nonlocal(self):
        return self.integral is not None

    @property
    def positive(self):
        return self.info()

    def build(self, symbol_name, symbol_name_nonlocal='g'):
        # Filters for the coefficients of the derivative terms
        reg1 = re.compile(r'^({0:s}|{1:s}){2:s}\d+$'.format(symbol_name.lower(), symbol_name.upper(), self.block)).search
        reg2 = re.compile(r'\d+$').search
        # Filter for the positivity information function
        reg3 = re.compile(r'^({0:s}|{1:s})info{2:s}'.format(symbol_name.lower(), symbol_name.upper(), self.block)).search
        # Filter for the integral terms
        reg4 = re.compile(r'^({0:s}|{1:s}){2:s}i'.format(symbol_name.lower(), symbol_name.upper(), self.block)).search

        # Filters for the coefficients of the derivative terms
        reg5 = re.compile(r'^({0:s}|{1:s}){2:s}\d+$'.format(symbol_name_nonlocal.lower(), symbol_name_nonlocal.upper(), self.block)).search
        # Filter for the positivity information function
        reg7 = re.compile(r'^({0:s}|{1:s})info{2:s}'.format(symbol_name_nonlocal.lower(), symbol_name_nonlocal.upper(), self.block)).search

        for symbol in self.ns.keys():
            fname = reg1(symbol)
            iname = reg3(symbol)
            jname = reg4(symbol)

            # Find non-local stuff
            gname  = reg5(symbol)
            giname = reg7(symbol)

            if fname is not None:
                fname = fname.group(0)

                # also grab the order
                try:
                    order = int(reg2(symbol).group(0)[-1])
                except Exception as e:
                    print('Failed to lookup order for symbol %s.' % symbol)

                if self.debug: print('LinBlock {0:s}.'.format(self.block), symbol)
                self.coeffs[order] = self.ns[symbol]
            elif gname is not None:
                gname = gname.group(0)

                # also grab the order
                try:
                    order = int(reg2(symbol).group(0)[-1])
                except Exception as e:
                    print('Failed to lookup order for symbol %s.' % symbol)

                if self.debug: print('LinBlock {0:s}.'.format(self.block), symbol)
                self.nlcoeffs[order] = self.ns[symbol]

            # Deal with the other two functions we might want to access
            elif iname is not None:
                if self.debug: print('LinBlock {0:s} info: '.format(self.block), symbol)
                self.info = self.ns[symbol]
            elif giname is not None:
                if self.debug: print('LinBlock {0:s} info: '.format(self.block), symbol)
                self.nl_info = self.ns[symbol]
            elif jname is not None:
                if self.debug: print('LinBlock {0:s} integral: '.format(self.block), symbol)
                self.integral = self.ns[symbol]

    def quasi(self, u, phi):
        new_coeffs = np.empty(len(self.coeffs), dtype=object)
        for i, coeff in enumerate(self.coeffs):
            new_coeffs[i] = coeff(*u, *phi).simplify(eps=self.eps)

        # No non-local support yet
        new_nl_coeffs = np.empty(1, dtype=object)

        # update the integral term
        new_integral = None
        if self.has_nonlocal:
            new_integral = self.integral(*u, *phi).simplify(eps=self.eps)

        return QuasiOpBlock(self.block, info=np.asarray(self.info()),
                            coeffs=new_coeffs, nl_coeffs=new_nl_coeffs,
                            nl_info=np.zeros_like(self.info()).astype(bool),
                            integral=new_integral)

    def __str__(self):
        return 'LinBlock[%s]' % self.block

    def __repr__(self):
        return "LinBlock[%s]" % self.block
