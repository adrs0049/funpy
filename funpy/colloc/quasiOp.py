#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np


class QuasiOp:
    def __init__(self, shape=None, *args, **kwargs):
        self.blocks = kwargs.pop('blocks', np.empty(shape, dtype=object))

    @property
    def shape(self):
        return self.blocks.shape

    def info(self):
        for i, j in np.ndindex(self.shape):
            print(self.blocks[i, j].info())

    def __getitem__(self, key):
        return self.blocks[key]

    def __setitem__(self, key, value):
        self.blocks[key] = value

    """ Mathematical operator support for QuasiMatrices """
    def __pos__(self):
        return self

    def __neg__(self):
        new_blocks = np.empty(self.shape, dtype=object)
        for i, j in np.ndindex(self.shape):
            new_blocks[i, j] = -self.blocks[i, j]

        return QuasiOp(blocks=new_blocks)

    def __add__(self, other):
        if not self.shape == other.shape:
            raise ValueError("Shape mismatch! %s != %s." % (self.shape, other.shape))

        new_blocks = np.empty(self.shape, dtype=object)
        for i, j in np.ndindex(self.shape):
            new_blocks[i, j] = self.blocks[i, j] + other.blocks[i, j]

        return QuasiOp(blocks=new_blocks)

    def __sub__(self, other):
        return self.__add__(-other)

    def __mul__(self, other):
        if not self.shape == other.shape:
            raise ValueError("Shape mismatch! %s != %s." % (self.shape, other.shape))

        new_blocks = np.empty(self.shape, dtype=object)
        for i, j in np.ndindex(self.shape):
            new_blocks[i, j] = self.blocks[i, j] + other.blocks[i, j]

        return QuasiOp(blocks=new_blocks)

    def __rmul__(self, other):
        return other * self
