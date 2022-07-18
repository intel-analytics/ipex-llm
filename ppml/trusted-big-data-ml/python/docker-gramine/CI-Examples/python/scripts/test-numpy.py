#!/usr/bin/env python3

import timeit

import numpy

try:
    import numpy.core._dotblas
except ImportError:
    pass

print("numpy version: " + numpy.__version__)

x = numpy.random.random((1000, 1000))

setup = "import numpy; x = numpy.random.random((1000, 1000))"
count = 5

t = timeit.Timer("numpy.dot(x, x.T)", setup=setup)
print("numpy.dot: " + str(t.timeit(count)/count) + " sec")
