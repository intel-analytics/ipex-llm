#!/usr/bin/env python3

import timeit

setup = """\
import numpy
import scipy.linalg as linalg
x = numpy.random.random((100,100))
z = numpy.dot(x, x.T)
"""
count = 5

t = timeit.Timer("linalg.cholesky(z, lower=True)", setup=setup)
print("linalg.cholesky: " + str(t.timeit(count)/count) + "sec")

t = timeit.Timer("linalg.svd(z)", setup=setup)
print("linalg.svd: " + str(t.timeit(count)/count) + "sec")
