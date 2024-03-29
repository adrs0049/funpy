#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np

from .bilinBlock import BiLinBlock
from .quasiOp import QuasiOp
from .source.support import execute_pycode


class BiLinOp:
    """ Represent a bilinear form """
    def __init__(self, ns, diffOrder, *args, **kwargs):
        # local namespace
        self.ns = ns
        # Should update this! from the positivity information!
        self.diffOrder = diffOrder
        self.blocks = kwargs.pop('blocks', None)

    def build(self, src, symbol_name, symbol_name_nonlocal='g', *args, **kwargs):
        debug = kwargs.get('debug', False)
        self.blocks = np.empty(src.shape, dtype=object)

        for i, j in np.ndindex(src.shape):
            if debug: print('Creating block [%d, %d].' % (i, j))
            execute_pycode(src[i, j], self.ns, debug=debug)
            self.blocks[i, j] = BiLinBlock(self.ns, '{0:d}{1:d}'.format(i, j),
                                           diff_order=self.diffOrder,
                                           debug=debug)

            # Build the block -> TODO exception handling!
            self.blocks[i, j].build(symbol_name, symbol_name_nonlocal)

    @property
    def shape(self):
        return self.blocks.shape

    def __getitem__(self, key):
        return self.blocks[key]

    def quasi(self, u, phi, *args, **kwargs):
        """
        Generates a linear quasi operator from the bilinear form, by assuming
        that one of the inputs (φ) to the bilinear form is fixed.
        """
        quasi = QuasiOp(self.shape)

        for i, j in np.ndindex(self.shape):
            quasi[i, j] = self.blocks[i, j].quasi(u, phi, *args, **kwargs)

        return quasi
