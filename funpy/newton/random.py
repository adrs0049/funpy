#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np

from ..fun import normalize, random


class RandomVector:
    def __init__(self, n, m, domain=[-1, 1], *args, **kwargs):

        # Needs to have a certain minimum size
        n = max(n, 8)

        # Generate some random functions
        self.B_u = normalize(random(n, m, domain=domain))
        self.B_l = np.random.rand(1).item()

        # Create the function
        self.C_u = normalize(random(n, m, domain=domain))
        self.C_l = np.random.rand(1).item()

        # Keep track of the other value we are tracking
        self.C_a_u = None
        self.C_a_l = None

        self.counter = 0
