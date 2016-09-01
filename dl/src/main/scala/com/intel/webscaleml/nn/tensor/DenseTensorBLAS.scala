package com.intel.webscaleml.nn.tensor

import com.github.fommil.netlib.BLAS.getInstance
import com.github.fommil.netlib.{BLAS, NativeSystemBLAS, F2jBLAS}
import com.intel.webscaleml.nn.tensor.TensorType.{DoubleReal => Real}
import com.intel.webscaleml.nn.tensor.TensorNumericMath._

object DenseTensorBLAS {
  private val blas =
    if(System.getProperty("BLAS") == "NativeSystemBLAS") {
      new NativeSystemBLAS
    } else if(System.getProperty("BLAS") == "F2JBLAS") {
      new F2jBLAS
    } else {
      getInstance
    }

  def getTensorBLAS: BLAS = blas

  var time = 0L

  def dgemm[@specialized(Float, Double) T](transa : String, transb : String, m : Int, n : Int, k : Int, alpha : T, a : Array[T], aOffset : Int,
      lda : Int, b : Array[T], bOffset : Int, ldb : Int, beta : T, c : Array[T], cOffset : Int, ldc : Int)(implicit ev:TensorNumeric[T]): Unit = {
    val _transa = (transa == "t" || transa == "T")
    val _transb = (transa == "t" || transa == "T")

    var _ldc = ldc
    if(n == 1)
      _ldc = m

    var _lda = lda
    if(_transa) {
      if(m == 1) {
        _lda = k
      }
    } else {
      if(k == 1) {
        _lda = m
      }
    }

    var _ldb = ldb
    if(_transb) {
      if(k == 1) {
        _ldb = n
      }
    } else {
      if(n == 1) {
        _ldb = k
      }
    }

    val start = System.nanoTime()
    ev.gemm(transa, transb, m, n, k, alpha, a, aOffset, _lda, b, bOffset, _ldb, beta, c, cOffset, _ldc)
    time += (System.nanoTime()-start)
  }

  def dgemv[@specialized(Float, Double) T](alpha : T, matrix : Tensor[T], vector : Tensor[T], beta : T, r : Tensor[T])(implicit ev:TensorNumeric[T]): Unit = {
    require(matrix.size(2) == vector.size(1), "matrix vector size doesn't match")
    require(matrix.size(1) == r.size(1), "matrix result size doesn't match")
    if(matrix.stride(1) == 1) {
      ev.gemv("N", matrix.size(1), matrix.size(2), alpha, matrix.storage().asInstanceOf[Storage[T]].array(), matrix.storageOffset() - 1,
        matrix.stride(2),vector.storage().asInstanceOf[Storage[T]].array(), vector.storageOffset() - 1, vector.stride(1), beta, r.storage().asInstanceOf[Storage[T]].array(),
        r.storageOffset() - 1, r.stride(1))
    } else if(matrix.stride(2) == 1) {
      ev.gemv("T", matrix.size(2), matrix.size(1), alpha, matrix.storage().asInstanceOf[Storage[T]].array(), matrix.storageOffset() - 1,
        matrix.stride(1),vector.storage().asInstanceOf[Storage[T]].array(), vector.storageOffset() - 1, vector.stride(1), beta, r.storage().asInstanceOf[Storage[T]].array(),
        r.storageOffset() - 1, r.stride(1))
    } else {
      val mat = matrix.contiguous()
      ev.gemv("T", mat.size(2), mat.size(1), alpha, mat.storage().asInstanceOf[Storage[T]].array(), mat.storageOffset() - 1,
        mat.stride(1),vector.storage().asInstanceOf[Storage[T]].array(), vector.storageOffset() - 1, vector.stride(1), beta, r.storage().asInstanceOf[Storage[T]].array(),
        r.storageOffset() - 1, r.stride(1))
    }
  }
}
