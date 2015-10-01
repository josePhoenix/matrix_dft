from distutils.core import setup
from Cython.Build import cythonize
from Cython.Compiler.Options import directive_defaults

directive_defaults['linetrace'] = True
directive_defaults['binding'] = True

import numpy as np

from distutils.extension import Extension

extensions = [
    Extension("matrix_dft", ["matrix_dft/backend_cython.pyx"],
        include_dirs = [np.get_include()],
        define_macros=[('CYTHON_TRACE', '1')]),
]
setup(
    include_dirs = [np.get_include()],
    ext_modules = cythonize(extensions)
)