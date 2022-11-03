""" Distributor init file

Distributors: you can add custom code here to support particular distributions
of numpy.

For example, this is a good place to put any checks for hardware requirements.

The numpy standard source distribution will not put code in this file, so you
can safely replace this file with your own version.
"""

try:
    import mkl
except ImportError:
    import warnings
    warnings.warn(
        "mkl-service package failed to import, therefore Intel(R) MKL "
        "initialization ensuring its correct out-of-the box operation under "
        "condition when Gnu OpenMP had already been loaded by Python process "
        "is not assured. Please install mkl-service package, see "
        "http://github.com/IntelPython/mkl-service", stacklevel=2)
