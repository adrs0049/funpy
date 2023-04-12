#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np


class Reflection:
    def __init__(self, f, *args, **kwargs):
        self.f = f
        self.domain = f.domain

    def _even_functor(self, x, n, i):
        return self.f(np.clip(n * (x - i / n), self.domain[0], self.domain[1]))

    def _odd_functor(self, x, n, i):
        return self.f(np.clip(n * (1. / n - (x - i / n)), self.domain[0], self.domain[1]))

    def __call__(self, x, n=1):
        if n == 1:
            return self.f(x)

        # Otherwise define the local tiles
        tile_bd = np.arange(0, n + 1, 1) / n

        values = np.empty((x.size, self.f.m), dtype=np.float)
        for k in range(1, tile_bd.size):
            mask = np.where((x >= tile_bd[k-1]) & (x <= tile_bd[k]))[0]
            values[mask, :] = self._odd_functor(x[mask], n, k-1) if k & 1 else self._even_functor(x[mask], n, k-1)

        return values
