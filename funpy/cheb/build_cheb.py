#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import subprocess, sys, os


def build_cheb_module():
    """
    Why are we doing this compile via a Makefile? This seems complicated.
    """
    dir_cwd = os.getcwd()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir_path)

    # get env variables
    d = dict(os.environ)

    python_exec = sys.executable
    make_process = subprocess.Popen('make clean && make all python_exec=%s' % python_exec,
                                    stdout=subprocess.PIPE, env=d,
                                    shell=True, stderr=subprocess.STDOUT)

    if make_process.wait() != 0:
        err_str = make_process.communicate()[0]
        raise NameError('Build of cheb module failed!\n{0:s}'.format(err_str.decode('utf-8')))

    # switch back to old dir
    os.chdir(dir_cwd)
