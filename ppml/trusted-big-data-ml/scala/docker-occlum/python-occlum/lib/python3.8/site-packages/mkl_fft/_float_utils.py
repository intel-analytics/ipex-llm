#!/usr/bin/env python
# Copyright (c) 2017-2019, Intel Corporation
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Intel Corporation nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from numpy import (half, float32, asarray, ndarray,
                   longdouble, float64, longcomplex, complex_)

__all__ = ['__upcast_float16_array', '__downcast_float128_array']

def __upcast_float16_array(x):
    """
    Used in _scipy_fft to upcast float16 to float32, 
    instead of float64, as mkl_fft would do"""
    if hasattr(x, "dtype"):
        xdt = x.dtype
        if xdt == half:
            # no half-precision routines, so convert to single precision
            return asarray(x, dtype=float32)
        if xdt == longdouble and not xdt == float64:
            raise ValueError("type %s is not supported" % xdt)
    if not isinstance(x, ndarray):
        __x = asarray(x)
        xdt = __x.dtype
        if xdt == half:
            # no half-precision routines, so convert to single precision
            return asarray(__x, dtype=float32)
        if xdt == longdouble and not xdt == float64:
            raise ValueError("type %s is not supported" % xdt)
        return __x
    return x


def __downcast_float128_array(x):
    """
    Used in _numpy_fft to unsafely downcast float128/complex256 to 
    complex128, instead of raising an error"""
    if hasattr(x, "dtype"):
        xdt = x.dtype
        if xdt == longdouble and not xdt == float64:
            return asarray(x, dtype=float64)
        elif xdt == longcomplex and not xdt == complex_:
            return asarray(x, dtype=complex_)
    if not isinstance(x, ndarray):
        __x = asarray(x)
        xdt = __x.dtype
        if xdt == longdouble and not xdt == float64:
            return asarray(x, dtype=float64)
        elif xdt == longcomplex and not xdt == complex_:
            return asarray(x, dtype=complex_)
        return __x
    return x
