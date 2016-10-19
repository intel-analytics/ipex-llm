/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.sparkdl.tensor

import com.github.fommil.netlib.BLAS.getInstance
import com.github.fommil.netlib.{BLAS, F2jBLAS, NativeSystemBLAS}
import com.intel.analytics.sparkdl.tensor.TensorNumericMath._

object DenseTensorBLAS {
  private val blas =
    if (System.getProperty("BLAS") == "NativeSystemBLAS") {
      new NativeSystemBLAS
    } else if (System.getProperty("BLAS") == "F2JBLAS") {
      new F2jBLAS
    } else {
      getInstance
    }

  def getTensorBLAS: BLAS = blas

  var time = 0L

  def gemm[@specialized(Float, Double) T](transa: String, transb: String,
    m: Int, n: Int, k: Int,
    alpha: T,
    a: Array[T], aOffset: Int, lda: Int,
    b: Array[T], bOffset: Int, ldb: Int,
    beta: T,
    c: Array[T], cOffset: Int, ldc: Int)(implicit ev: TensorNumeric[T]): Unit = {

    val _transa = (transa == "t" || transa == "T")
    val _transb = (transb == "t" || transb == "T")

    var _ldc = ldc
    if (n == 1) {
      _ldc = m
    }

    var _lda = lda
    if (_transa) {
      if (m == 1) {
        _lda = k
      }
    } else {
      if (k == 1) {
        _lda = m
      }
    }

    var _ldb = ldb
    if (_transb) {
      if (k == 1) {
        _ldb = n
      }
    } else {
      if (n == 1) {
        _ldb = k
      }
    }

    val start = System.nanoTime()
    ev.gemm(transa, transb, m, n, k, alpha, a, aOffset, _lda, b, bOffset, _ldb, beta, c,
      cOffset, _ldc)
    time += (System.nanoTime() - start)
  }

  def gemv[@specialized(Float, Double) T](alpha: T, matrix: Tensor[T], vector: Tensor[T],
    beta: T, r: Tensor[T])(implicit ev: TensorNumeric[T]): Unit = {

    require(matrix.size(2) == vector.size(1), "matrix vector size doesn't match")
    require(matrix.size(1) == r.size(1), "matrix result size doesn't match")
    if (matrix.stride(1) == 1) {
      ev.gemv("N", matrix.size(1), matrix.size(2), alpha, matrix.storage().array(),
        matrix.storageOffset() - 1,
        matrix.stride(2), vector.storage().array(), vector.storageOffset() - 1, vector.stride(1),
        beta, r.storage().array(),
        r.storageOffset() - 1, r.stride(1))
    } else if (matrix.stride(2) == 1) {
      ev.gemv("T", matrix.size(2), matrix.size(1), alpha, matrix.storage().array(),
        matrix.storageOffset() - 1,
        matrix.stride(1), vector.storage().array(), vector.storageOffset() - 1,
        vector.stride(1), beta, r.storage().array(),
        r.storageOffset() - 1, r.stride(1))
    } else {
      val mat = matrix.contiguous()
      ev.gemv("T", mat.size(2), mat.size(1), alpha, mat.storage().array(), mat.storageOffset() - 1,
        mat.stride(1), vector.storage().array(), vector.storageOffset() - 1, vector.stride(1),
        beta, r.storage().array(),
        r.storageOffset() - 1, r.stride(1))
    }
  }
}
