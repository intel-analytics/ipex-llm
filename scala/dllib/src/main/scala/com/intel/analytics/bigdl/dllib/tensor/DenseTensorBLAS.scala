/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath._

object DenseTensorBLAS {
  var time = 0L

  /**
   * The gemm routines compute a scalar-matrix-matrix product and
   * add the result to a scalar-matrix product, with general matrices.
   * C := alpha*op(A)*op(B) + beta*C,
   * where:
   * op(X) is one of op(X) = X, or op(X) = XT,
   * alpha and beta are scalars,
   * A, B and C are matrices:
   * op(A) is an m-by-k matrix,
   * op(B) is a k-by-n matrix,
   * C is an m-by-n matrix.
   *
   * this interface treat the input array as column-major array.
   *
   * @param transa Specifies the form of op(A) used in the matrix multiplication:
   *               if transa=CblasNoTrans, then op(A) = A;
                   if transa=CblasTrans, then op(A) = AT;
   * @param transb Specifies the form of op(B) used in the matrix multiplication:
                   if transb=CblasNoTrans, then op(B) = B;
                   if transb=CblasTrans, then op(B) = BT;
   * @param m Specifies the number of rows of the matrix op(A) and of the matrix C.
   *          The value of m must be at least zero.
   * @param n Specifies the number of columns of the matrix op(B) and the number of
   *          columns of the matrix C. The value of n must be at least zero.
   * @param k Specifies the number of columns of the matrix op(A) and the number of
   *          rows of the matrix op(B). The value of k must be at least zero.
   * @param alpha Specifies the scalar alpha.
   * @param a  Array. if transa=CblasNoTrans, size lda*k. if transa=CblasTrans, size lda*m.
   * @param aOffset a offset
   * @param lda Specifies the leading dimension of a as declared in the calling (sub)program.
   *            if transa=CblasNoTrans, lda must be at least max(1, m).
   *            if transa=CblasTrans, lda must be at least max(1, k).
   * @param b Array. if transb=CblasNoTrans, size ldb by n. if transb=CblasTrans, size ldb by k.
   * @param bOffset b offset
   * @param ldb Specifies the leading dimension of b as declared in the calling (sub)program.
   *            if transb=CblasNoTrans, ldb must be at least max(1, m).
   *            if transb=CblasTrans, ldb must be at least max(1, k).
   * @param beta Specifies the scalar beta.
                 When beta is equal to zero, then c need not be set on input.
   * @param c Array, size ldc by n. Before entry, the leading m-by-n part of the array c must
   *          contain the matrix C, except when beta is equal to zero,
   *          in which case c need not be set on entry.
   * @param cOffset c offset
   * @param ldc ldc must be at least max(1, m).
   * @param ev
   * @tparam T
   */
  def gemm[@specialized(Float, Double) T](transa: Char, transb: Char,
    m: Int, n: Int, k: Int,
    alpha: T,
    a: Array[T], aOffset: Int, lda: Int,
    b: Array[T], bOffset: Int, ldb: Int,
    beta: T,
    c: Array[T], cOffset: Int, ldc: Int)(implicit ev: TensorNumeric[T]): Unit = {

    val _transa = (transa == 't' || transa == 'T')
    val _transb = (transb == 't' || transb == 'T')

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

  /**
   * The gemv routines perform a matrix-vector operation defined as
   * y := alpha*A*x + beta*y,
   * or
   * y := alpha*A'*x + beta*y,
   * where:
   * alpha and beta are scalars,
   * x and y are vectors,
   * A is an m-by-n matrix.
   *
   * this interface treat the input array as column-major array.
   *
   * @param trans Specifies the operation:
   *              if trans=CblasNoTrans, then y := alpha*A*x + beta*y;
   *              if trans=CblasTrans, then y := alpha*A'*x + beta*y;
   *
   * @param m Specifies the number of rows of the matrix A.
   *          The value of m must be at least zero.
   * @param n Specifies the number of columns of the matrix A.
   *          The value of n must be at least zero.
   * @param alpha Specifies the scalar alpha.
   * @param a Array, size lda* n. Before entry, the leading m-by-n part of the array a must
   *          contain the matrix A.
   * @param aOffset a offset
   * @param lda Specifies the leading dimension of a as declared in the calling (sub)program.
   *            the value of lda must be at least max(1, m).
   * @param x Array, size at least (1+(n-1)*abs(incx)).
   *          Before entry, the incremented array x must contain the vector x.
   * @param xOffset x offset
   * @param incx Specifies the increment for the elements of x.
   *             The value of incx must not be zero.
   * @param beta Specifies the scalar beta.
   *             When beta is set to zero, then y need not be set on input.
   * @param y Array, size at least (1 +(m - 1)*abs(incy)). Before entry with non-zero beta,
   *          the incremented array y must contain the vector y.
   * @param yOffset y offset
   * @param incy Specifies the increment for the elements of y.
   *             The value of incy must not be zero.
   * @param ev
   * @tparam T
   */
  def gemv[@specialized(Float, Double) T](trans: Char, m: Int, n: Int,
    alpha: T,
    a: Array[T], aOffset: Int, lda: Int,
    x: Array[T], xOffset: Int, incx: Int,
    beta: T,
    y: Array[T], yOffset: Int, incy: Int)(implicit ev: TensorNumeric[T]): Unit = {

    val start = System.nanoTime()
    ev.gemv(trans, m, n, alpha, a, aOffset, lda, x, xOffset, incx, beta, y,
      yOffset, incy)
    time += (System.nanoTime() - start)
  }

  /**
   * The ger routines perform a matrix-vector operation defined as
   * A := alpha*x*y'+ A,
   * where:
   * alpha is a scalar,
   * x is an m-element vector,
   * y is an n-element vector,
   * A is an m-by-n general matrix.
   *
   * this interface treat the input array as column-major array.
   *
   * @param m Specifies the number of rows of the matrix A.
   *          The value of m must be at least zero.
   * @param n Specifies the number of columns of the matrix A.
   *          The value of n must be at least zero.
   * @param alpha Specifies the scalar alpha.
   * @param x Array, size at least (1 + (m - 1)*abs(incx)).
   *          Before entry, the incremented array x must contain the m-element vector x.
   * @param xOffset x offset
   * @param incx Specifies the increment for the elements of x.
   *             The value of incx must not be zero.
   * @param y Array, size at least (1 + (n - 1)*abs(incy)).
   *          Before entry, the incremented array y must contain the n-element vector y.
   * @param yOffset y offset
   * @param incy Specifies the increment for the elements of y.
   *             The value of incy must not be zero.
   * @param a Array, size lda * n. Before entry,
   *          the leading m-by-n part of the array a must contain the matrix A.
   * @param aOffset a offset
   * @param lda Specifies the leading dimension of a as declared in the calling (sub)program.
   *            the value of lda must be at least max(1, m).
   * @param ev
   * @tparam T
   */
  def ger[@specialized(Float, Double) T](m: Int, n: Int,
    alpha: T,
    x: Array[T], xOffset: Int, incx: Int,
    y: Array[T], yOffset: Int, incy: Int,
    a: Array[T], aOffset: Int, lda: Int)(implicit ev: TensorNumeric[T]): Unit = {

    val start = System.nanoTime()
    ev.ger(m, n, alpha, x, xOffset, incx, y, yOffset, incy, a, aOffset, lda)
    time += (System.nanoTime() - start)
  }
  /**
   * to be fixed: this interface treat the input tensor as row-major array.
   * @param alpha
   * @param matrix
   * @param vector
   * @param beta
   * @param r
   * @param ev
   * @tparam T
   */
  def gemv[@specialized(Float, Double) T](alpha: T, matrix: Tensor[T], vector: Tensor[T],
    beta: T, r: Tensor[T])(implicit ev: TensorNumeric[T]): Unit = {
    require(matrix.size(2) == vector.size(1), "matrix vector size doesn't match")
    require(matrix.size(1) == r.size(1), "matrix result size doesn't match")
    if (matrix.stride(1) == 1) {
      ev.gemv('N', matrix.size(1), matrix.size(2), alpha, matrix.storage().array(),
        matrix.storageOffset() - 1,
        matrix.stride(2), vector.storage().array(), vector.storageOffset() - 1, vector.stride(1),
        beta, r.storage().array(),
        r.storageOffset() - 1, r.stride(1))
    } else if (matrix.stride(2) == 1) {
      ev.gemv('T', matrix.size(2), matrix.size(1), alpha, matrix.storage().array(),
        matrix.storageOffset() - 1,
        matrix.stride(1), vector.storage().array(), vector.storageOffset() - 1,
        vector.stride(1), beta, r.storage().array(),
        r.storageOffset() - 1, r.stride(1))
    } else {
      val mat = matrix.contiguous()
      ev.gemv('T', mat.size(2), mat.size(1), alpha, mat.storage().array(), mat.storageOffset() - 1,
        mat.stride(1), vector.storage().array(), vector.storageOffset() - 1, vector.stride(1),
        beta, r.storage().array(),
        r.storageOffset() - 1, r.stride(1))
    }
  }
}
