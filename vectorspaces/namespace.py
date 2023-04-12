#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen

class Namespace(dict):
    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            self[k] = v

    @classmethod
    def from_hdf5(cls, hdf5_file):
        return cls(**hdf5_file.attrs)

    def writeHDF5(self, fh):
        for k, v in self.items():
            fh.attrs[k] = v

    def readHDF5(self, fh, *args, **kwargs):
        for k, v in fh.attrs.items():
            self[k] = v

    def __str__(self):
        rstr = 'Namespace:'
        for k, v in self.items():
            rstr += '{} '.format(str(v))
        return rstr

    def __repr__(self):
        return self.__str__()
