#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np

class Mapping(object):
    """ Class mapping [-1, 1] to arbitrary domains. """
    def __init__(self, ends, *args, **kwargs):
        # TODO deal with unbounded case
        self.ends = ends
        self.fwd = kwargs.pop('fwd', None)
        self.der = kwargs.pop('der', None)
        self.bwd = kwargs.pop('bwd', None)

        if self.fwd is None or self.der is None or self.bwd is None:
            self.__linear(ends)

    def __linear(self, ends):
        """ Creates a linear map structure.

            ends is a two vector with the domain ends.

            fwd -> maps [-1, 1] to [ends[0], ends[1]]
            der -> is the derivative of the map defined in for
            bwd -> is the inverse map
        """
        self.fwd = lambda y: (self.ends[1] * (y + 1) + self.ends[0] * (1 - y)) / 2
        self.der = lambda y: np.diff(self.ends) / 2 + 0*y
        self.bwd = lambda x: (2 * x - self.ends[0] - self.ends[1]) / np.diff(self.ends)

    def __call__(self, x):
        return self.fwd(x)

    def __repr__(self):
        return f"{self.__class__.__name__}(ends={self.ends})"

    def __str__(self):
        return 'Map([%.2f, %.2f] -> [-1, 1])' % (self.ends[0], self.ends[1])

    def inv(self):
        return Mapping(fwd=self.inv, der=lambda x: 1./self.der(x), inv=self.fwd)
