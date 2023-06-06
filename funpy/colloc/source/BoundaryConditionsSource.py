#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
from .gen import CodeGeneratorBackend


class BoundaryConditionsSource:
    def __init__(self, *args, **kwargs):
        self.ops = []

        # Reference to the parent
        self.op = kwargs.pop('op')

        # info symbol_name
        self.symbol_name = kwargs.pop('symbol_name', 'bc')

    def append(self, value):
        self.ops.append(value)

    def emit(self):
        # code gen
        cg = CodeGeneratorBackend()
        cg.begin(tab=4*" ")
        cg.write(45 * '#')
        cg.write('# Constraints')
        cg.write(45 * '#')

        for i, op in enumerate(self.ops):
            symbol_name = '{0:s}{1:d}'.format(self.symbol_name, i)
            op.symbol_name = symbol_name
            op.emit(cg, symbol_name,
                    self.op.func_names, self.op.cfunc_names,
                    self.op.ftype, self.op.domain,
                    allow_unknown_functions=True, no_evaluation=False)

        return cg.end()

    def finish(self):
        pass

    def __getitem__(self, idx):
        assert self.ops is not None, 'Have you called finish?'

        try:
            src = self.ops[idx]
        except Exception as e:
            raise e
        return src

    def __iter__(self):
        self.ipos = 0
        return self

    def __next__(self):
        if self.ipos >= len(self.ops):
            raise StopIteration
        self.ipos += 1
        return self[self.ipos-1]

    def __repr__(self):
        rstr = '\t-> ' + type(self).__name__ + ':\n'
        for op in self.ops:
            rstr += '\t\t-> ' + repr(op) + '\n'
        return rstr
