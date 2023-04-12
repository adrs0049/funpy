import numpy as np
from distutils.core import Extension, setup
from Cython.Build import cythonize

sourcefiles = ['detail.pyx'] # , 'ufuncs.pyx']
compileargs = ['-O3', '-march=native']

setup(
    ext_modules=cythonize([
        Extension("detail",
                  sourcefiles,
                  include_dirs=[np.get_include()],
                  extra_compile_args=compileargs),
        Extension("ufuncs",
                  ['ufuncs.pyx'],
                  include_dirs=[np.get_include()],
                  extra_compile_args=compileargs)
    ])
)
