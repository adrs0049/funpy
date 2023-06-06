#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import os
import numpy as np
import h5py as h5
from numpy.testing import assert_, assert_raises, assert_almost_equal

from funpy.vectorspaces.namespace import Namespace


class TestNamespace:

    def test_io(self):
        ns = Namespace(a=1.0, b=2.0)

        fname = 'funpy_namespace_test.h5'
        hdf5_file = h5.File(fname, 'w')
        ns.writeHDF5(hdf5_file)
        hdf5_file.close()

        # Read the file
        hdf5_file = h5.File(fname, 'r')
        g = Namespace.from_hdf5(hdf5_file)

        assert_(g['a'] == ns['a'])
        assert_(g['b'] == ns['b'])
        os.remove(fname)
