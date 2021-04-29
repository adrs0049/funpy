#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np

from colloc.linBlock import LinBlock
from colloc.quasiOp import QuasiOp
from colloc.tools import execute_pycode


class LinOp:
    def __init__(self, ns, diffOrder, *args, **kwargs):
        # local namespace
        self.ns = ns
        self.diffOrder = diffOrder
        self.blocks = kwargs.pop('blocks', None)

    def build(self, src, symbol_name, symbol_name_nonlocal='g', *args, **kwargs):
        debug = kwargs.get('debug', False)
        self.blocks = np.empty(src.shape, dtype=object)

        for i, j in np.ndindex(src.shape):
            if debug: print('Creating block [%d, %d].' % (i, j))

            # Compile the required python code.
            pycode = src[i, j].emit()

            # Execute the compiled code in the local namespace
            execute_pycode(pycode, self.ns, debug=debug)

            # Finally create linear block representation
            self.blocks[i, j] = LinBlock(self.ns, '{0:d}{1:d}'.format(i, j), debug=debug)

            # Build the block -> TODO exception handling!
            self.blocks[i, j].build(src[i, j])

    @property
    def shape(self):
        return self.blocks.shape

    def __getitem__(self, key):
        return self.blocks[key]

    def quasi(self, u, *args, **kwargs):
        quasi = QuasiOp(self.shape)

        for i, j in np.ndindex(self.shape):
            quasi[i, j] = self.blocks[i, j].quasi(u, *args, **kwargs)

        return quasi
