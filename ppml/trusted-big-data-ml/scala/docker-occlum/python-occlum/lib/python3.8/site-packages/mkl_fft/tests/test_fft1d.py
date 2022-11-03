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

from __future__ import division, absolute_import, print_function

import numpy as np
from numpy.testing import (
        TestCase, run_module_suite, assert_, assert_raises, assert_equal,
        assert_warns, assert_allclose)
from numpy import random as rnd
import sys
import warnings

import mkl_fft

def naive_fft1d(vec):
    L = len(vec)
    phase = -2j*np.pi*(np.arange(L)/float(L))
    phase = np.arange(L).reshape(-1, 1) * phase
    return np.sum(vec*np.exp(phase), axis=1)


def _datacopied(arr, original):
    """
    Strict check for `arr` not sharing any data with `original`,
    under the assumption that arr = asarray(original)
    """
    if arr is original:
        return False
    if not isinstance(original, np.ndarray) and hasattr(original, '__array__'):
        return False
    return arr.base is None


class Test_mklfft_vector(TestCase):
    def setUp(self):
        rnd.seed(1234567)
        self.xd1 = rnd.standard_normal(128)
        self.xf1 = self.xd1.astype(np.float32)
        self.xz1 = rnd.standard_normal((128,2)).view(dtype=np.complex128).squeeze()
        self.xc1 = self.xz1.astype(np.complex64)

    def test_vector1(self):
        """check that mkl_fft gives the same result of numpy.fft"""
        f1 = mkl_fft.fft(self.xz1)
        f2 = naive_fft1d(self.xz1)
        assert_allclose(f1,f2, rtol=1e-7, atol=2e-12)

        f1 = mkl_fft.fft(self.xc1)
        f2 = naive_fft1d(self.xc1)
        assert_allclose(f1,f2, rtol=2e-6, atol=2e-6)

    def test_vector2(self):
        "ifft(fft(x)) is identity"
        f1 = mkl_fft.fft(self.xz1)
        f2 = mkl_fft.ifft(f1)
        assert_(np.allclose(self.xz1,f2))

        f1 = mkl_fft.fft(self.xc1)
        f2 = mkl_fft.ifft(f1)
        assert_( np.allclose(self.xc1,f2))

        f1 = mkl_fft.fft(self.xd1)
        f2 = mkl_fft.ifft(f1)
        assert_( np.allclose(self.xd1,f2))

        f1 = mkl_fft.fft(self.xf1)
        f2 = mkl_fft.ifft(f1)
        assert_( np.allclose(self.xf1,f2, atol = 2.0e-7))

    def test_vector3(self):
        "fft(ifft(x)) is identity"
        f1 = mkl_fft.ifft(self.xz1)
        f2 = mkl_fft.fft(f1)
        assert_(np.allclose(self.xz1,f2))

        f1 = mkl_fft.ifft(self.xc1)
        f2 = mkl_fft.fft(f1)
        assert_( np.allclose(self.xc1,f2))

        f1 = mkl_fft.ifft(self.xd1)
        f2 = mkl_fft.fft(f1)
        assert_( np.allclose(self.xd1,f2))

        f1 = mkl_fft.ifft(self.xf1)
        f2 = mkl_fft.fft(f1)
        assert_( np.allclose(self.xf1, f2, atol = 2.0e-7))

    def test_vector4(self):
        "fft of strided is same as fft of contiguous copy"
        x = self.xz1[::2]
        f1 = mkl_fft.fft(x)
        f2 = mkl_fft.fft(x.copy())
        assert_(np.allclose(f1,f2))

        x = self.xz1[::-1]
        f1 = mkl_fft.fft(x)
        f2 = mkl_fft.fft(x.copy())
        assert_(np.allclose(f1,f2))

    def test_vector5(self):
        "fft in-place is the same as fft out-of-place"
        x = self.xz1.copy()[::-2]
        f1 = mkl_fft.fft(x, overwrite_x=True)
        f2 = mkl_fft.fft(self.xz1[::-2])
        assert_(np.allclose(f1,f2))
    
    def test_vector6(self):
        "fft in place"
        x = self.xz1.copy()
        f1 = mkl_fft.fft(x, overwrite_x=True)
        assert_(not _datacopied(f1, x))  # this is in-place

        x = self.xz1.copy()
        f1 = mkl_fft.fft(x[::-2], overwrite_x=True)
        assert_( not np.allclose(x, self.xz1) ) # this is also in-place
        assert_( np.allclose(x[-2::-2], self.xz1[-2::-2]) ) 
        assert_( np.allclose(x[-1::-2], f1) ) 

    def test_vector7(self):
        "fft of real array is the same as fft of its complex cast"
        x = self.xd1[3:17:2]
        f1 = mkl_fft.fft(x)
        f2 = mkl_fft.fft(x.astype(np.complex128))
        assert_(np.allclose(f1,f2))

    def test_vector8(self):
        "ifft of real array is the same as fft of its complex cast"
        x = self.xd1[3:17:2]
        f1 = mkl_fft.ifft(x)
        f2 = mkl_fft.ifft(x.astype(np.complex128))
        assert_(np.allclose(f1,f2))

    def test_vector9(self):
        "works on subtypes of ndarray"
        mask = np.zeros(self.xd1.shape, dtype='int')
        mask[1] = 1
        mask[-2] = 1
        x = np.ma.masked_array(self.xd1, mask=mask)
        f1 = mkl_fft.fft(x)
        f2 = mkl_fft.fft(self.xd1)
        assert_allclose(f1, f2)

    def test_vector10(self):
        "check n for real arrays"
        x = self.xd1[:8].copy()
        f1 = mkl_fft.fft(x, n = 7)
        f2 = mkl_fft.fft(self.xd1[:7])
        assert_allclose(f1, f2)

        f1 = mkl_fft.fft(x, n = 9)
        y = self.xd1[:9].copy()
        y[-1] = 0.0
        f2 = mkl_fft.fft(y)
        assert_allclose(f1, f2)

    def test_vector11(self):
        "check n for complex arrays"
        x = self.xz1[:8].copy()
        f1 = mkl_fft.fft(x, n = 7)
        f2 = mkl_fft.fft(self.xz1[:7])
        assert_allclose(f1, f2)

        f1 = mkl_fft.fft(x, n = 9)
        y = self.xz1[:9].copy()
        y[-1] = 0.0 + 0.0j
        f2 = mkl_fft.fft(y)
        assert_allclose(f1, f2)

    def test_vector12(self):
        "check fft of float-valued array"
        x = np.arange(20)
        f1 = mkl_fft.fft(x)
        f2 = mkl_fft.fft(x.astype(np.float64))
        assert_allclose(f1, f2)


class DuckArray(np.ndarray): pass

class Test_mklfft_matrix(TestCase):
    def setUp(self):
        rnd.seed(1234567)
        self.ad2 = rnd.standard_normal((4, 3))
        self.af2 = self.ad2.astype(np.float32)
        self.az2 = np.dot(
              rnd.standard_normal((17, 15, 2)),
              np.array([1.0 + 0.0j, 0.0 + 1.0j], dtype=np.complex128)
        )
        self.ac2 = self.az2.astype(np.complex64)
        self.mat = self.az2.view(DuckArray)
        self.xd1 = rnd.standard_normal(128)

    def test_matrix1(self):
        x = self.az2.copy()
        f1 = mkl_fft.fft(x)
        f2 = np.array([ mkl_fft.fft(x[i]) for i in range(x.shape[0])])
        assert_allclose(f1, f2)

        f1 = mkl_fft.fft(x, axis=0)
        f2 = np.array([ mkl_fft.fft(x[:, i]) for i in range(x.shape[1])]).T
        assert_allclose(f1, f2)

    def test_matrix2(self):
        f1 = mkl_fft.fft(self.az2)
        f2 = mkl_fft.fft(self.mat)
        assert_allclose(f1, f2)

    def test_matrix3(self):
        x = self.az2.copy()
        f1 = mkl_fft.fft(x[::3,::-1])
        f2 = mkl_fft.fft(x[::3,::-1].copy())
        assert_allclose(f1, f2)

    def test_matrix4(self):
        x = self.az2.copy()
        f1 = mkl_fft.fft(x[::3,::-1])
        f2 = mkl_fft.fft(x[::3,::-1], overwrite_x=True)
        assert_allclose(f1, f2)

    def test_matrix5(self):
        x = self.ad2;
        f1 = mkl_fft.fft(x)
        f2 = mkl_fft.ifft(f1)
        assert_allclose(x, f2, atol=1e-10)

    def test_matrix6(self):
        x = self.ad2;
        f1 = mkl_fft.ifft(x)
        f2 = mkl_fft.fft(f1)
        assert_allclose(x, f2, atol=1e-10)

    def test_matrix7(self):
        x = self.ad2.copy()
        f1 = mkl_fft.fft(x)
        f2 = np.array([ mkl_fft.fft(x[i]) for i in range(x.shape[0])])
        assert_allclose(f1, f2)

        f1 = mkl_fft.fft(x, axis=0)
        f2 = np.array([ mkl_fft.fft(x[:, i]) for i in range(x.shape[1])]).T
        assert_allclose(f1, f2)

    def test_matrix8(self):
        from numpy.lib.stride_tricks import as_strided
        x = self.xd1[:10].copy()
        y = as_strided(x, shape=(4,4,), strides=(2*x.itemsize, x.itemsize))
        f1 = mkl_fft.fft(y)
        f2 = mkl_fft.fft(y.copy())
        assert_allclose(f1, f2, atol=1e-15, rtol=1e-7)


class Test_mklfft_rank3(TestCase):
    def setUp(self):
        rnd.seed(1234567)
        self.ad3 = rnd.standard_normal((7, 11, 19))
        self.af3 = self.ad3.astype(np.float32)
        self.az3 = np.dot(
              rnd.standard_normal((17, 13, 15, 2)),
              np.array([1.0 + 0.0j, 0.0 + 1.0j], dtype=np.complex128)
        )
        self.ac3 = self.az3.astype(np.complex64)

    def test_array1(self):
        x = self.az3
        for ax in range(x.ndim):
            f1 = mkl_fft.fft(x, axis = ax)
            f2 = mkl_fft.ifft(f1, axis = ax)
            assert_allclose(f2, x, atol=2e-15)

    def test_array2(self):
        x = self.ad3
        for ax in range(x.ndim):
            f1 = mkl_fft.fft(x, axis = ax)
            f2 = mkl_fft.ifft(f1, axis = ax)
            assert_allclose(f2, x, atol=2e-15)

    def test_array3(self):
        x = self.az3
        for ax in range(x.ndim):
            f1 = mkl_fft.ifft(x, axis = ax)
            f2 = mkl_fft.fft(f1, axis = ax)
            assert_allclose(f2, x, atol=2e-15)

    def test_array4(self):
        x = self.ad3
        for ax in range(x.ndim):
            f1 = mkl_fft.ifft(x, axis = ax)
            f2 = mkl_fft.fft(f1, axis = ax)
            assert_allclose(f2, x, atol=2e-15)


    def test_array5(self):
        """Inputs with zero strides are handled correctly"""
        z = self.az3
        z1 = z[np.newaxis]
        f1 = mkl_fft.fft(z1, axis=-1)
        f2 = mkl_fft.fft(z1.reshape(z1.shape), axis=-1)
        assert_allclose(f1, f2, atol=2e-15)
        z1 = z[:, np.newaxis]
        f1 = mkl_fft.fft(z1, axis=-1)
        f2 = mkl_fft.fft(z1.reshape(z1.shape), axis=-1)
        assert_allclose(f1, f2, atol=2e-15)
        z1 = z[:, :, np.newaxis]
        f1 = mkl_fft.fft(z1, axis=-1)
        f2 = mkl_fft.fft(z1.reshape(z1.shape), axis=-1)
        assert_allclose(f1, f2, atol=2e-15)
        z1 = z[:, :, :, np.newaxis]
        f1 = mkl_fft.fft(z1, axis=-1)
        f2 = mkl_fft.fft(z1.reshape(z1.shape), axis=-1)
        assert_allclose(f1, f2, atol=2e-15)

    def test_array6(self):
        """Inputs with Fortran layout are handled correctly, issue 29"""
        z = self.az3
        z = z.astype(z.dtype, order='F')
        y1 = mkl_fft.fft(z, axis=0)
        y2 = mkl_fft.fft(self.az3, axis=0)
        assert_allclose(y1, y2, atol=2e-15)
        y1 = mkl_fft.fft(z, axis=-1)
        y2 = mkl_fft.fft(self.az3, axis=-1)
        assert_allclose(y1, y2, atol=2e-15)


class Test_mklfft_rfft(TestCase):
    def setUp(self):
        rnd.seed(1234567)
        self.v1 = rnd.randn(16)
        self.m2 = rnd.randn(5,7)
        self.t3 = rnd.randn(5,7,11)

    def test1(self):
        x = self.v1.copy()
        f1 = mkl_fft.rfft(x)
        f2 = mkl_fft.irfft(f1)
        assert_allclose(f2,x)

    def test2(self):
        x = self.v1.copy()
        f1 = mkl_fft.irfft(x)
        f2 = mkl_fft.rfft(f1)
        assert_allclose(f2,x)

    def test3(self):
        for a in range(0,2):
            for ovwr_x in [True, False]:
                for dt, atol in zip([np.float32, np.float64], [2e-7, 2e-15]):
                    x = self.m2.copy().astype(dt)
                    f1 = mkl_fft.rfft(x, axis=a, overwrite_x=ovwr_x)
                    f2 = mkl_fft.irfft(f1, axis=a, overwrite_x=ovwr_x)
                    assert_allclose(f2, self.m2.astype(dt), atol=atol, err_msg=(a, ovwr_x))

    def test4(self):
        for a in range(0,2):
            for ovwr_x in [True, False]:
                for dt, atol in zip([np.float32, np.float64], [2e-7, 2e-15]):
                    x = self.m2.copy().astype(dt)
                    f1 = mkl_fft.irfft(x, axis=a, overwrite_x=ovwr_x)
                    f2 = mkl_fft.rfft(f1, axis=a, overwrite_x=ovwr_x)
                    assert_allclose(f2, self.m2.astype(dt), atol=atol)

    def test5(self):
        for a in range(0,3):
            for ovwr_x in [True, False]:
                for dt, atol in zip([np.float32, np.float64], [4e-7, 4e-15]):
                    x = self.t3.copy().astype(dt)
                    f1 = mkl_fft.irfft(x, axis=a, overwrite_x=ovwr_x)
                    f2 = mkl_fft.rfft(f1, axis=a, overwrite_x=ovwr_x)
                    assert_allclose(f2, self.t3.astype(dt), atol=atol)

if __name__ == "__main__":
    run_module_suite(argv = sys.argv)
