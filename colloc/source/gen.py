#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
#
# https://gist.github.com/kafagy/4be733ea943fcb2f89feb7378438b6a9
#
from __future__ import print_function, absolute_import, division

import string

class CodeGeneratorBackend:

    def begin(self, tab="\t"):
        self.code = []
        self.tab = tab
        self.level = 0

    def end(self):
        return "".join(self.code)

    def write(self, string, end='\n', debug=False):
        # if string is empty don't print random white-spaces
        if string == '':
            self.code.append(end)
            return

        if debug and (len(self.code) % 10) == 0:
            self.code.append(self.tab * self.level + '# Line: {0}'.format(len(self.code)) + end)

        self.code.append(self.tab * self.level + string + end)

    def indent(self):
        self.level = self.level + 1

    def dedent(self):
        if self.level == 0:
            raise SyntaxError
        self.level = self.level - 1
