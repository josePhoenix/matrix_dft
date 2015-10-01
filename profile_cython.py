#!/usr/bin/env python
# encoding: utf-8
# filename: profile.py

import pstats, cProfile

# from matrix_dft.tests import test_backend_cython
import numpy as np

import os
from matrix_dft import backend_cython

matrix_x = np.random.randn(1024, 1024) + np.exp(-2.0 * np.pi * 1.0j * np.random.randn(1024, 1024))
matrix_y = np.random.randn(1024, 1024) + np.exp(-2.0 * np.pi * 1.0j * np.random.randn(1024, 1024))

print 'generated test matrices'

cProfile.runctx("backend_cython._matrix_multiply(matrix_x, matrix_y)", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()