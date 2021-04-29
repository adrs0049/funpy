#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np

from ac.gen import CodeGeneratorBackend

from colloc.source.DiffOperatorSource import DiffOperatorSource
from colloc.source.IntegralOperatorSource import IntegralOperatorSource


class OperatorSourceBlock:
    def __init__(self, i, j, *args, **kwargs):
        self.pos = [i, j]
        self.dops = []
        self.iops = []

        # Reference to the parent
        self.op = kwargs.pop('op')

        # To be built once done
        self.ops = None

        # Count number of Diff operators
        self.dcount = 0

        # Count number of Integral operators
        self.icount = 0

        # info symbol_name
        self.symbol_name = ''

    @property
    def diffOrder(self):
        diffOrder = 0
        for op in self.dops:
            diffOrder = max(op.order, diffOrder)
        return diffOrder

    @property
    def numNonLocal(self):
        return self.icount

    @property
    def numDiffOps(self):
        return self.dcount

    @property
    def expr(self):
        expr = self.ops[0].expr
        for i in range(1, len(self.ops)):
            expr += self.ops[i].expr

        return expr

    def append(self, value):
        value.pos = self.pos

        if isinstance(value, DiffOperatorSource):
            self.dcount += 1
            self.dops.append(value)
        elif isinstance(value, IntegralOperatorSource):
            value.offset = self.icount
            self.icount += 1
            self.iops.append(value)
        else:
            raise RuntimeError('Unknown operator source \"{0:s}\"!'.
                               format(type(value).__name__))

    def emit(self):
        # code gen
        cg = CodeGeneratorBackend()
        cg.begin(tab=4*" ")
        cg.write(45 * '#')
        cg.write('# Derivative block {0:s}[{1:d}, {2:d}].'\
                 .format(self.op.name, self.pos[0], self.pos[1]))
        cg.write(45 * '#')

        for op in self.ops:
            op.emit(cg, self.op.name, self.op.func_names,
                    self.op.cfunc_names,
                    self.op.ftype, self.op.domain,
                    allow_unknown_functions=True, no_evaluation=True)

        # Generate positivity information for the block
        self.emit_posInfo(cg, self.op.name)
        return cg.end()

    def emit_posInfo(self, cg, name):
        positivity_information = self.posInfo()

        # Store the symbol name for the info
        self.symbol_name = '{0:s}{1:s}{2:d}{3:d}'.format(name, 'info', self.pos[0], self.pos[1])

        # create code for positivity information
        cg.write('')
        cg.write('# Return sorted by derivative term order, followed by the integral terms.')
        cg.write('def {0:s}():'.format(self.symbol_name))
        cg.indent()
        cg.write('return [{0}]'.format(', '.join(map(str, positivity_information))))
        cg.dedent()

    def finish(self):
        # Sort derivatives by their order from lowest to highest
        self.dops.sort(key=lambda x: x.order)
        self.ops = self.dops + self.iops

    def posInfo(self):
        positivity_information = np.ones(len(self.dops) + 1).astype(bool)
        for i, op in enumerate(self.dops):
            positivity_information[i] = op.posInfo

        posInfo = False
        for op in self.iops:
            posInfo |= op.posInfo

        positivity_information[-1] = posInfo
        return positivity_information

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
        rstr = '\t-> ' + type(self).__name__ + ' ({0:d}, {1:d})'.\
                format(self.pos[0], self.pos[1]) + ':\n'
        for op in self.ops:
            rstr += '\t\t-> ' + repr(op) + '\n'
        return rstr
