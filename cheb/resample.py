#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen


class Resample(object):
    def __init__(self, op, *args, **kwargs):
        self.minSamples = kwargs.pop('minSamples', 3)
        self.op = op
        self.values = None

    def get_n(self, f):
        if f.size == 0:
            n = 2**np.ceil(np.log2(self.minSamples - 1)) + 1
        else:
            pow = np.log2(f.shape[0] - 1)
            if pow == np.floor(pow) and pow > 5:
                n = np.round(2**(np.floor(pow) + 0.5)) + 1
                n = n - np.remainder(n, 2) + 1
            else:
                n = 2**(np.floor(pow) + 1) + 1
        # TODO: make sure that n is not too large
        return n

class Resample1(Resample):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._call = self.__nested

    def __call__(self, f):
        return self._call(f)

    def __resample(self, f):
        n = self.get_n(f)

        # update f-values
        f.prolong(n)
        v1 = polyval(f.coeffs)

        # compute the new values
        self.values = self.op(v1)
        return self.values

    def __nested(self, f):
        if self.values is None:
            self.__resample(f)
        else:
            n = 2 * f.shape[0] - 1

            # check that n is too large
            f.prolong(n)
            fvalues = polyval(f.coeffs)
            v1 = fvalues[1:-1:2, :]

            # shift the values
            self.values[0:n:2, :] = self.values

            # compute the new values
            self.values[1:-1:2, :] = self.op(v1)
        return self.values

class Resample2(Resample):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._call = self.__nested

    def __call__(self, f, g):
        return self._call(f, g)

    def __resample(self, f, g):
        n = self.get_n(f)

        # update f-values
        f.prolong(n)
        v1 = polyval(f.coeffs)
        g.prolong(n)
        v2 = polyval(g.coeffs)

        # compute the new values
        self.values = self.op(v1, v2)
        return self.values

    def __nested(self, f, g):
        if self.values is None:
            self.__resample(f)
        else:
            n = 2 * f.shape[0] - 1

            # check that n is too large
            f.prolong(n)
            fvalues = polyval(f.coeffs)
            v1 = fvalues[1:-1:2, :]
            g.prolong(n)
            gvalues = polyval(g.coeffs)
            v2 = gvalues[1:-1:2, :]

            # shift the values
            self.values[0:n:2, :] = self.values

            # compute the new values
            self.values[1:-1:2, :] = self.op(v1, v2)
        return values
