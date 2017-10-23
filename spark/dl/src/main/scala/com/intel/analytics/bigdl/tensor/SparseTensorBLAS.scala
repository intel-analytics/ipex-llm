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

object SparseTensorBLAS {

  /**
   * Perform r := beta * r + alpha * mat * vec
   * mat should be a 2D SparseTensor, vec should be a 1D DenseTensor,
   * r should be a 2D DenseTensor.
   *
   * @param alpha alpha
   * @param mat a 2D SparseTensor
   * @param vec a 1D DenseTensor
   * @param beta beta
   * @param r result, 2D DenseTensor
   * @param ev tensor numeric
   * @tparam T numeric type
   */
  def coomv[@specialized(Float, Double) T](
        alpha: T,
        mat: Tensor[T],
        vec: Tensor[T],
        beta: T,
        r: Tensor[T])(implicit ev: TensorNumeric[T]): Unit = {
    (alpha, mat, vec, beta, r)  match {
      case (alpha: Double, a: SparseTensor[Double], x: DenseTensor[Double],
      beta: Double, y: DenseTensor[Double]) =>
        dcoomv(alpha, a, x, beta, y)
      case (alpha: Float, a: SparseTensor[Float], x: DenseTensor[Float],
      beta: Float, y: DenseTensor[Float]) =>
        scoomv(alpha, a, x, beta, y)
      case _ =>
        throw new IllegalArgumentException(s"Sparse addmv doesn't support")
    }
  }

  private def scoomv(
        alpha: Float,
        A: SparseTensor[Float],
        x: DenseTensor[Float],
        beta: Float,
        y: DenseTensor[Float]): Unit = {
    val xValues = x.storage().array()
    val xOffset = x.storageOffset() - 1
    val yValues = y.storage().array()
    val yOffset = y.storageOffset() - 1
    val mA: Int = A._shape(0)
    val nA: Int = A._shape(1)

    val Avals = A._values.array()
    val AstorageOffset = A.storageOffset() - 1
    val Arows = A._indices(0)
    val ArowOffset = A._indicesOffset(0)
    val Acols = A._indices(1)
    val AcolOffset = A._indicesOffset(1)

    if (beta != 1.0) {
      y.mul(beta)
    }
    // Perform matrix-vector multiplication and add to y
    var valueCounter = 0
    while (valueCounter < A.nElement()) {
      val Arow = Arows(valueCounter + AstorageOffset) - ArowOffset
      val Acol = Acols(valueCounter + AstorageOffset) - AcolOffset
      val Aval = Avals(valueCounter + AstorageOffset)
      yValues(Arow + yOffset) += Aval * alpha * xValues(Acol + xOffset)
      valueCounter += 1
    }
  }

  private def dcoomv(
        alpha: Double,
        A: SparseTensor[Double],
        x: DenseTensor[Double],
        beta: Double,
        y: DenseTensor[Double]): Unit = {
    val xValues = x.storage().array()
    val xOffset = x.storageOffset() - 1
    val yValues = y.storage().array()
    val yOffset = y.storageOffset() - 1
    val mA: Int = A._shape(0)
    val nA: Int = A._shape(1)

    val Avals = A._values.array()
    val AstorageOffset = A.storageOffset() - 1
    val Arows = A._indices(0)
    val ArowOffset = A._indicesOffset(0)
    val Acols = A._indices(1)
    val AcolOffset = A._indicesOffset(1)

    if (beta != 1.0) {
      y.mul(beta)
    }
    // Perform matrix-vector multiplication and add to y
    var valueCounter = 0
    while (valueCounter < A.nElement()) {
      val Arow = Arows(valueCounter + AstorageOffset) - ArowOffset
      val Acol = Acols(valueCounter + AstorageOffset) - AcolOffset
      val Aval = Avals(valueCounter + AstorageOffset)
      yValues(Arow + yOffset) += Aval * alpha * xValues(Acol + xOffset)
      valueCounter += 1
    }
  }

  def coomm[@specialized(Float, Double) T](
        alpha: T,
        mat1: Tensor[T],
        mat2: Tensor[T],
        beta: T,
        r: Tensor[T])(implicit ev: TensorNumeric[T]): Unit = {
    (alpha, mat1, mat2, beta, r)  match {
      case (alpha: Float, a: SparseTensor[Float], x: DenseTensor[Float],
            beta: Float, y: DenseTensor[Float]) =>
        scoomm(alpha, a, x, beta, y)
      case (alpha: Double, a: SparseTensor[Double], x: DenseTensor[Double],
            beta: Double, y: DenseTensor[Double]) =>
        dcoomm(alpha, a, x, beta, y)
      case (alpha: Float, a: DenseTensor[Float], x: SparseTensor[Float],
            beta: Float, y: DenseTensor[Float]) =>
        scoomm(alpha, a, x, beta, y)
      case (alpha: Double, a: DenseTensor[Double], x: SparseTensor[Double],
            beta: Double, y: DenseTensor[Double]) =>
        dcoomm(alpha, a, x, beta, y)
      case _ =>
        throw new IllegalArgumentException(s"Sparse addmm doesn't support")
    }
  }

  private def scoomm(
        alpha: Float,
        A: SparseTensor[Float],
        B: DenseTensor[Float],
        beta: Float,
        C: DenseTensor[Float]): Unit = {
    val mA: Int = A._shape(0)
    val nB: Int = B.size(2)
    val kA: Int = A._shape(1)
    val kB: Int = B.size(1)

    val Avals = A._values.array()
    val AstorageOffset = A.storageOffset() - 1
    val ArowIndices = A._indices(0)
    val ArowOffset = A._indicesOffset(0)
    val AcolIndices = A._indices(1)
    val AcolOffset = A._indicesOffset(1)

    val Bvals = B.storage().array()
    val bOffset = B.storageOffset() - 1
    val Cvals = C.storage().array()
    val cOffset = C.storageOffset() - 1

    // Scale matrix first if `beta` is not equal to 1.0
    if (beta != 1.0) {
      C.mul(beta)
    }
    // Perform matrix multiplication and add to C. The rows of A are multiplied by the columns of
    // B, and added to C.
    var index = 0
    if (B.stride(2) == 1 && B.size(2) == B.stride(1)) {
      while (index < A.nElement()) {
        val curMA = ArowIndices(index + AstorageOffset) - ArowOffset
        val curKA = AcolIndices(index + AstorageOffset) - AcolOffset
        var n = 0
        while (n < nB) {
          Cvals(curMA * nB + n) += alpha * Avals(index + AstorageOffset) *
            Bvals(curKA * nB + n + bOffset)
          n += 1
        }
        index += 1
      }
    } else {
      while (index < A.nElement()) {
        val curMA = ArowIndices(index + AstorageOffset) - ArowOffset
        val curKA = AcolIndices(index + AstorageOffset) - AcolOffset
        var n = 0
        while (n < nB) {
          Cvals(curMA * nB + n + cOffset) += alpha * Avals(index + AstorageOffset) *
            Bvals(curKA + n * kB + bOffset)
          n += 1
        }
        index += 1
      }

    }
  }

  private def dcoomm(
        alpha: Double,
        A: SparseTensor[Double],
        B: DenseTensor[Double],
        beta: Double,
        C: DenseTensor[Double]): Unit = {
    val mA: Int = A._shape(0)
    val nB: Int = B.size(2)
    val kA: Int = A._shape(1)
    val kB: Int = B.size(1)

    val Avals = A._values.array()
    val AstorageOffset = A.storageOffset() - 1
    val ArowIndices = A._indices(0)
    val ArowOffset = A._indicesOffset(0)
    val AcolIndices = A._indices(1)
    val AcolOffset = A._indicesOffset(1)

    val Bvals = B.storage().array()
    val bOffset = B.storageOffset() - 1
    val Cvals = C.storage().array()
    val cOffset = C.storageOffset() - 1

    // Scale matrix first if `beta` is not equal to 1.0
    if (beta != 1.0) {
      C.mul(beta)
    }
    // Perform matrix multiplication and add to C. The rows of A are multiplied by the columns of
    // B, and added to C.
    var index = 0
    if (B.stride(2) == 1 && B.size(2) == B.stride(1)) {
      while (index < A.nElement()) {
        val curMA = ArowIndices(index + AstorageOffset) - ArowOffset
        val curKA = AcolIndices(index + AstorageOffset) - AcolOffset
        var n = 0
        while (n < nB) {
          Cvals(curMA * nB + n) += alpha * Avals(index + AstorageOffset) *
            Bvals(curKA * nB + n + bOffset)
          n += 1
        }
        index += 1
      }
    } else {
      while (index < A.nElement()) {
        val curMA = ArowIndices(index + AstorageOffset) - ArowOffset
        val curKA = AcolIndices(index + AstorageOffset) - AcolOffset
        var n = 0
        while (n < nB) {
          Cvals(curMA * nB + n + cOffset) += alpha * Avals(index + AstorageOffset) *
            Bvals(curKA + n * kB + bOffset)
          n += 1
        }
        index += 1
      }

    }
  }

  private def scoomm(
        alpha: Float,
        A: DenseTensor[Float],
        B: SparseTensor[Float],
        beta: Float,
        C: DenseTensor[Float]): Unit = {
    val kB: Int = B.size(1)
    val nB: Int = B.size(2)
    val mA: Int = A.size(1)
    val kA: Int = A.size(2)

    val Avals = A.storage().array()
    val aOffset = A.storageOffset() - 1
    val Cvals = C.storage().array()
    val cOffset = C.storageOffset() - 1

    val Bvals = B._values.array()
    val BstorageOffset = B.storageOffset() - 1
    val BrowIndices = B._indices(0)
    val BrowIndicesOffset = B._indicesOffset(0)
    val BcolIndices = B._indices(1)
    val BcolIndicesOffset = B._indicesOffset(1)

    // Scale matrix first if `beta` is not equal to 1.0
    if (beta != 1.0) {
      C.mul(beta)
    }
    // Perform matrix multiplication and add to C. The rows of B are multiplied by the columns of
    // A, and added to C.
    var index = 0
    if (A.stride(2) == 1 && A.size(2) == A.stride(1)) {
      while (index < B.nElement()) {
        val curKB = BrowIndices(index + BstorageOffset) - BrowIndicesOffset
        val curNB = BcolIndices(index + BstorageOffset) - BcolIndicesOffset
        var n = 0
        while (n < mA) {
          Cvals(n * nB + curNB + cOffset) += alpha * Bvals(index + BstorageOffset) *
            Avals(n * kA + curKB + aOffset)
          n += 1
        }
        index += 1
      }
    } else {
      while (index < B.nElement()) {
        val curKB = BrowIndices(index + BstorageOffset) - BrowIndicesOffset
        val curNB = BcolIndices(index + BstorageOffset) - BcolIndicesOffset
        var n = 0
        while (n < mA) {
          Cvals(n * nB + curNB + cOffset) += alpha * Bvals(index + BstorageOffset) *
            Avals(n + mA * curKB + aOffset)
          n += 1
        }
        index += 1
      }
    }
  }

  private def dcoomm(
        alpha: Double,
        A: DenseTensor[Double],
        B: SparseTensor[Double],
        beta: Double,
        C: DenseTensor[Double]): Unit = {
    val kB: Int = B.size(1)
    val nB: Int = B.size(2)
    val mA: Int = A.size(1)
    val kA: Int = A.size(2)

    val Avals = A.storage().array()
    val aOffset = A.storageOffset() - 1
    val Cvals = C.storage().array()
    val cOffset = C.storageOffset() - 1

    val Bvals = B._values.array()
    val BstorageOffset = B.storageOffset() - 1
    val BrowIndices = B._indices(0)
    val BrowIndicesOffset = B._indicesOffset(0)
    val BcolIndices = B._indices(1)
    val BcolIndicesOffset = B._indicesOffset(1)

    // Scale matrix first if `beta` is not equal to 1.0
    if (beta != 1.0) {
      C.mul(beta)
    }
    // Perform matrix multiplication and add to C. The rows of B are multiplied by the columns of
    // A, and added to C.
    var index = 0
    if (A.stride(2) == 1 && A.size(2) == A.stride(1)) {
      while (index < B.nElement()) {
        val curKB = BrowIndices(index + BstorageOffset) - BrowIndicesOffset
        val curNB = BcolIndices(index + BstorageOffset) - BcolIndicesOffset
        var n = 0
        while (n < mA) {
          Cvals(n * nB + curNB + cOffset) += alpha * Bvals(index + BstorageOffset) *
            Avals(n * kA + curKB + aOffset)
          n += 1
        }
        index += 1
      }
    } else {
      while (index < B.nElement()) {
        val curKB = BrowIndices(index + BstorageOffset) - BrowIndicesOffset
        val curNB = BcolIndices(index + BstorageOffset) - BcolIndicesOffset
        var n = 0
        while (n < mA) {
          Cvals(n * nB + curNB + cOffset) += alpha * Bvals(index + BstorageOffset) *
            Avals(n + mA * curKB + aOffset)
          n += 1
        }
        index += 1
      }
    }
  }
}
