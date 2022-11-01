#!/usr/bin/env python3

import sys;
print(sys.path);

import timeit
import numpy
import pandas

print("pandas version: " + pandas.__version__)

df_random = pandas.DataFrame(numpy.random.randint(100, size=(10, 10)),
                             columns=list('ABCDEFGHIJ'))

print("Random Dataframe:")
print(df_random)

