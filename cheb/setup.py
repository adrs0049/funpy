import numpy as np
from distutils.core import Extension, setup
from Cython.Build import cythonize

detail = Extension(name="detail",
                    sources=["detail.pyx"],
                    include_dirs=[np.get_include()],
                    extra_compile_args=['-O3', '-march=native'],
                    )

setup(
    ext_modules=cythonize(detail),
)

ufunc = Extension(name="ufuncs",
                  sources=["ufuncs.pyx"],
                  include_dirs=[np.get_include()],
                  extra_compile_args=['-O3', '-march=native'],
                )
setup(
    ext_modules=cythonize(ufunc),
)
