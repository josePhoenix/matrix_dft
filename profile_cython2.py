import line_profiler
import numpy as np
from matrix_dft import backend_cython

matrix_x = np.random.randn(1024, 1024) + np.exp(-2.0 * np.pi * 1.0j * np.random.randn(1024, 1024))
matrix_y = np.random.randn(1024, 1024) + np.exp(-2.0 * np.pi * 1.0j * np.random.randn(1024, 1024))

print 'generated test matrices'

def do_multiply(a, b):
    backend_cython._matrix_multiply(a, b)

profile = line_profiler.LineProfiler(backend_cython._matrix_multiply)
profile.runcall(backend_cython._matrix_multiply, matrix_x, matrix_y)
profile.print_stats()