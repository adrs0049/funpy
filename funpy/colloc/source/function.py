#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
from sympy import Function


class DummyFunction:
    def __init__(self, name, dummy):
        self.name = name
        self.dummy = dummy
        self.expr = Function(name)(dummy)
