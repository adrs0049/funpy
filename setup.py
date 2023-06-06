#!/usr/bin/env python
import numpy as np
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

compileargs = ['-O3', '-march=native', '-fno-fast-math']

extensions = [
        Extension(
                'funpy.cheb.detail',
                ["funpy/cheb/detail.pyx"],
                include_dirs=[np.get_include()],
                extra_compile_args=compileargs,
        ),

        Extension("funpy.cheb.ufuncs",
                  ['funpy/cheb/ufuncs.pyx'],
                  include_dirs=[np.get_include()],
                  extra_compile_args=compileargs
        ),
]

pkgs = find_packages()
print(pkgs)

setup(
    name='funpy',
    version='0.1.0',
    description='funPy',
    author='Andreas Buttenschoen',
    author_email='andreas@buttenschoen.ca',
    url='https://github.com/adrs0049/funpy',
    test_suite='tests',
    packages=find_packages(),
    package_data={'': ['*.pxd', '*.pyx']},
    ext_modules = cythonize(extensions),
    install_requires=[],
    license='BSD 3-clause',
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
)
