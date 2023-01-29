#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Roughly based on: http://stackoverflow.com/questions/11443302/compiling-numpy-with-openblas-integration

from __future__ import print_function

import numpy as np
from time import time
import click

@click.command()
@click.option('--size', default=4096)
@click.option('--env', default='native')
@click.option('--type', default='int')
def  main(size, env, type):
    if type == 'int':
        benchmark_int(size, env)
    if type == 'float':
        benchmark_float(size, env)

def benchmark_int(size, env):
    f = open('/ppml/tests/numpy/benchmark_' + env + '_' + str(size) + '_int', 'w')
    # Let's take the randomness out of random numbers (for reproducibility)
    np.random.seed(0)

    A, B = np.random.randint(100, size=(size, size)), np.random.randint(100, size=(size, size))
    E = np.random.randint(100, size=(int(size), int(size )))
    F = np.random.randint(100, size=(int(size * 5), int(size * 5)))
    F = np.dot(F, F.T)
    G = np.random.randint(100, size=(int(size), int(size)))

    # Matrix multiplication
    N = 5
    t = time()
    for i in range(N):
        np.dot(A, B)
    delta = time() - t
    print('Dotted two %dx%d matrices in %0.2f s.' % (size, size, delta / N))
    f.write('Dotted two %dx%d matrices in %0.2f s.\n' % (size, size, delta / N))
    del A, B

    # Singular Value Decomposition (SVD)
    N = 5
    t = time()
    for i in range(N):
        np.linalg.svd(E, full_matrices = False)
    delta = time() - t
    print("SVD of a %dx%d matrix in %0.2f s." % (size, size, delta / N))
    f.write("SVD of a %dx%d matrix in %0.2f s.\n" % (size, size, delta / N))
    del E

    # Cholesky Decomposition
    N = 5
    t = time()
    for i in range(N):
        np.linalg.cholesky(F)
    delta = time() - t
    print("Cholesky decomposition of a %dx%d matrix in %0.2f s." % (size * 5, size * 5, delta / N))
    f.write("Cholesky decomposition of a %dx%d matrix in %0.2f s.\n" % (size * 5, size * 5, delta / N))

    # Eigendecomposition
    t = time()
    for i in range(N):
        np.linalg.eig(G)
    delta = time() - t
    print("Eigendecomposition of a %dx%d matrix in %0.2f s." % (size, size, delta / N))
    f.write("Eigendecomposition of a %dx%d matrix in %0.2f s.\n" % (size, size, delta / N))
    f.close()

def benchmark_float(size, env):
    f = open('/ppml/tests/numpy/benchmark_' + env + '_' + str(size) + '_float', 'w')
    # Let's take the randomness out of random numbers (for reproducibility)
    np.random.seed(0)

    A, B = np.random.random((size, size)), np.random.random((size, size))
    E = np.random.random((int(size / 2), int(size / 4)))
    F = np.random.random((int(size), int(size)))
    F = np.dot(F, F.T)
    G = np.random.random((int(size / 2), int(size / 2)))

    # Matrix multiplication
    N = 5
    t = time()
    for i in range(N):
        np.dot(A, B)
    delta = time() - t
    print('Dotted two %dx%d matrices in %0.2f s.' % (size, size, delta / N))
    f.write('Dotted two %dx%d matrices in %0.2f s.\n' % (size, size, delta / N))
    del A, B

    # Singular Value Decomposition (SVD)
    N = 5
    t = time()
    for i in range(N):
        np.linalg.svd(E, full_matrices = False)
    delta = time() - t
    print("SVD of a %dx%d matrix in %0.2f s." % (size / 2, size / 4, delta / N))
    f.write("SVD of a %dx%d matrix in %0.2f s.\n" % (size / 2, size / 4, delta / N))
    del E

    # Cholesky Decomposition
    N = 5
    t = time()
    for i in range(N):
        np.linalg.cholesky(F)
    delta = time() - t
    print("Cholesky decomposition of a %dx%d matrix in %0.2f s." % (size, size, delta / N))
    f.write("Cholesky decomposition of a %dx%d matrix in %0.2f s.\n" % (size, size, delta / N))

    # Eigendecomposition
    t = time()
    for i in range(N):
        np.linalg.eig(G)
    delta = time() - t
    print("Eigendecomposition of a %dx%d matrix in %0.2f s." % (size / 2, size / 2, delta / N))
    f.write("Eigendecomposition of a %dx%d matrix in %0.2f s.\n" % (size / 2, size / 2, delta / N))
    f.close()

if __name__ == '__main__':
    size = 16384
    A, B = np.random.random((size, size)), np.random.random((size, size))
    np.dot(A, B)
    main()

