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

import java.util

import com.intel.analytics.bigdl.mkl.MKL
import com.intel.analytics.bigdl.utils.RandomGenerator._

/**
 * class of math operation
 */
class TensorNumericMath

/**
 * Math operation for tensor
 */
object TensorNumericMath {

  type NumericWildCard = Any
  /**
   * This type is used to denote that the numeric type of tensor is not restricted.
   * The use-case usually is used to do some tensor operations when we do not make sure
   * their concrete types, but they must have the same type.
   *
   * For example if we want to copy tensor1 from tensor2, and we only know they are
   * the same type tensor without the information about their concrete type.
   *
   * We can use the following code:
   *
   * `tensor1.asInstanceOf[Tensor[NumericWildcard]]
   * .copy(tensor2.asInstanceOf[Tensor[NumericWildcard]])`
   */
  type NumericWildcard = Any

  /**
   * define tensor math operation
   */
  trait TensorNumeric[@specialized(Float, Double) T] extends Serializable {
    def one: T = fromType[Int](1)

    def zero: T = fromType[Int](0)

    def plus(x: T, y: T): T

    def minus(x: T, y: T): T

    def times(x: T, y: T): T

    def divide(x: T, y: T): T

    def exp(x: T): T

    def log(x: T): T

    def max(x: T, y: T): T

    def min(x: T, y: T): T

    def sqrt(x: T): T

    def tanh(x: T): T

    def abs(x: T): T

    def or(x: T, y: T): T

    def and(x: T, y: T): T

    def negative(x: T): T

    def pow(x: T): T

    def pow(x: T, y: T): T

    def log1p(x: T): T

    def isGreater(x: T, y: T): Boolean

    def isGreaterEq(x: T, y: T): Boolean

    def rand(): T

    def randn(): T

    def gemm(transa: Char, transb: Char, m: Int, n: Int, k: Int, alpha: T, a: Array[T],
      aOffset: Int, lda: Int, b: Array[T], bOffset: Int, ldb: Int,
      beta: T, c: Array[T], cOffset: Int, ldc: Int)

    def gemv(trans: Char, m: Int, n: Int, alpha: T, a: Array[T], aoffset: Int, lda: Int,
      x: Array[T], xOffset: Int, incx: Int, beta: T, y: Array[T], yOffset: Int, incy: Int)

    def axpy(n: Int, da: T, dx: Array[T], _dx_offset: Int, incx: Int, dy: Array[T],
      _dy_offset: Int, incy: Int)

    def dot(n: Int, dx: Array[T], _dx_offset: Int, incx: Int, dy: Array[T], _dy_offset: Int,
      incy: Int): T

    def ger(m: Int, n: Int, alpha: T, x: Array[T], _x_offset: Int, incx: Int, y: Array[T],
      _y_offset: Int,
      incy: Int, a: Array[T], _a_offset: Int, lda: Int)

    def fill(data: Array[T], fromIndex: Int, toIndex: Int, value: T): Unit

    def fromType[K](k: K)(implicit c: ConvertableFrom[K]): T

    def toType[K](t: T)(implicit c: ConvertableTo[K]): K

    def vPowx(n: Int, a: Array[T], aOffset: Int, b: T, y: Array[T], yOffset: Int): Unit

    def vLn(n: Int, a: Array[T], aOffset: Int, y: Array[T], yOffset: Int): Unit

    def vExp(n: Int, a: Array[T], aOffset: Int, y: Array[T], yOffset: Int): Unit

    def vSqrt(n: Int, a: Array[T], aOffset: Int, y: Array[T], yOffset: Int): Unit

    def vTanh(n: Int, a: Array[T], aOffset: Int, y: Array[T], yOffset: Int): Unit

    def vAbs(n: Int, a: Array[T], aOffset: Int, y: Array[T], yOffset: Int): Unit

    def vLog1p(n: Int, a: Array[T], aOffset: Int, y: Array[T], yOffset: Int): Unit

    def scal(n: Int, sa: T, sx: Array[T], offset: Int, incx: Int): Unit

    def inv(v: T): T

    def erf(v: T): T

    def erfc(v: T): T

    def logGamma(v: T): T

    def digamma(v: T): T

    def add(n: Int, a: Array[T], offset: Int, v: T, stride: Int): Unit

    def sub(n: Int, a: Array[T], offset: Int, v: T, stride: Int): Unit

    def vAdd(n: Int, a: Array[T], aOffset: Int, b: Array[T], bOffset: Int, y: Array[T],
      yOffset: Int): Unit

    def vSub(n: Int, a: Array[T], aOffset: Int, b: Array[T], bOffset: Int, y: Array[T],
      yOffset: Int): Unit

    def vMul(n: Int, a: Array[T], aOffset: Int, b: Array[T], bOffset: Int, y: Array[T],
      yOffset: Int): Unit

    def vDiv(n: Int, a: Array[T], aOffset: Int, b: Array[T], bOffset: Int, y: Array[T],
      yOffset: Int): Unit

    def sum(n: Int, a: Array[T], aOffset: Int, stride: Int): T

    def prod(n: Int, a: Array[T], aOffset: Int, stride: Int): T

    def arraycopy(src: Array[T], srcPos: Int,
                  dest: Array[T], destPos: Int, length: Int): Unit

    def getType(): TensorDataType

    def addcmul(value: T, n: Int,
      self: Array[T], selfOffset: Int,
      a: Array[T], aOffset: Int,
      b: Array[T], bOffset: Int): Unit

    def addcdiv(value: T, n: Int,
      self: Array[T], selfOffset: Int,
      a: Array[T], aOffset: Int,
      b: Array[T], bOffset: Int): Unit

    def nearlyEqual(a: T, b: T, epsilon: Double): Boolean

    def floor(a: T): T

    def ceil(a: T): T

    def isFinite(a: T): Boolean

    def isNan(a: T): Boolean

    def isInf(a: T): Boolean

    def round(a: T): T

    def truncate(a: T): T

    def floorDiv(a: T, b: T): T

    def clip(a: T, lower: T, upper: T): T
  }

  /**
   * define tensor math operation
   */
  abstract class UndefinedTensorNumeric[@specialized(Float, Double) T](typeName: String)
    extends TensorNumeric[T] {
    def plus(x: T, y: T): T =
      throw new UnsupportedOperationException(typeName
        + " in tensor does not support plus operation")

    def minus(x: T, y: T): T =
      throw new UnsupportedOperationException(typeName
        + " in tensor does not support minus operation")

    def times(x: T, y: T): T =
      throw new UnsupportedOperationException(typeName
        + " in tensor does not support times operation")

    def divide(x: T, y: T): T =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support divide operation")

    def exp(x: T): T =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support exp operation")

    def prod(n: Int, a: Array[T], aOffset: Int, stride: Int): T =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support prod operation")

    def log(x: T): T =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support log operation")

    def max(x: T, y: T): T =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support max operation")

    def min(x: T, y: T): T =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support min operation")

    def sqrt(x: T): T =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support sqrt operation")

    def tanh(x: T): T =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support tanh operation")

    def abs(x: T): T =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support abs operation")

    def or(x: T, y: T): T =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support or operation")

    def and(x: T, y: T): T =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support and operation")

    def negative(x: T): T =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support negative operation")

    def pow(x: T): T =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support pow operation")

    def pow(x: T, y: T): T =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support pow operation")

    def log1p(x: T): T =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support log1p operation")

    def isGreater(x: T, y: T): Boolean =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support isGreater operation")

    def isGreaterEq(x: T, y: T): Boolean =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support isGreaterEq operation")

    def rand(): T =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support rand operation")

    def randn(): T =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support randn operation")

    def gemm(transa: Char, transb: Char, m: Int, n: Int, k: Int, alpha: T, a: Array[T],
      aOffset: Int, lda: Int, b: Array[T], bOffset: Int, ldb: Int,
      beta: T, c: Array[T], cOffset: Int, ldc: Int): Unit =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support gemm operation")

    def gemv(trans: Char, m: Int, n: Int, alpha: T, a: Array[T], aoffset: Int, lda: Int,
      x: Array[T], xOffset: Int, incx: Int, beta: T, y: Array[T], yOffset: Int, incy: Int): Unit =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support gemv operation")

    def axpy(n: Int, da: T, dx: Array[T], _dx_offset: Int, incx: Int, dy: Array[T],
      _dy_offset: Int, incy: Int): Unit =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support axpy operation")

    def dot(n: Int, dx: Array[T], _dx_offset: Int, incx: Int, dy: Array[T], _dy_offset: Int,
      incy: Int): T =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support dot operation")

    def ger(m: Int, n: Int, alpha: T, x: Array[T], _x_offset: Int, incx: Int, y: Array[T],
      _y_offset: Int,
      incy: Int, a: Array[T], _a_offset: Int, lda: Int): Unit =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support ger operation")

    def fill(data: Array[T], fromIndex: Int, toIndex: Int, value: T): Unit =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support fill operation")

    def fromType[K](k: K)(implicit c: ConvertableFrom[K]): T =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support fromType operation")

    def toType[K](t: T)(implicit c: ConvertableTo[K]): K =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support toType operation")

    def vPowx(n: Int, a: Array[T], aOffset: Int, b: T, y: Array[T], yOffset: Int): Unit =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support vPowx operation")

    def vLn(n: Int, a: Array[T], aOffset: Int, y: Array[T], yOffset: Int): Unit =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support vLn operation")

    def vExp(n: Int, a: Array[T], aOffset: Int, y: Array[T], yOffset: Int): Unit =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support vLn operation")

    def vSqrt(n: Int, a: Array[T], aOffset: Int, y: Array[T], yOffset: Int): Unit =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support vSqrt operation")

    def vTanh(n: Int, a: Array[T], aOffset: Int, y: Array[T], yOffset: Int): Unit =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support vTanh operation")

    def vAbs(n: Int, a: Array[T], aOffset: Int, y: Array[T], yOffset: Int): Unit =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support vAbs operation")

    def vLog1p(n: Int, a: Array[T], aOffset: Int, y: Array[T], yOffset: Int): Unit =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support vLog1p operation")

    def scal(n: Int, sa: T, sx: Array[T], offset: Int, incx: Int): Unit =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support scal operation")

    def inv(v: T): T =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support inv operation")

    def add(n: Int, a: Array[T], offset: Int, v: T, stride: Int): Unit =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support add operation")

    def sub(n: Int, a: Array[T], offset: Int, v: T, stride: Int): Unit =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support sub operation")

    def vAdd(n: Int, a: Array[T], aOffset: Int, b: Array[T], bOffset: Int, y: Array[T],
      yOffset: Int): Unit =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support vAdd operation")

    def vSub(n: Int, a: Array[T], aOffset: Int, b: Array[T], bOffset: Int, y: Array[T],
      yOffset: Int): Unit =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support vSub operation")

    def vMul(n: Int, a: Array[T], aOffset: Int, b: Array[T], bOffset: Int, y: Array[T],
      yOffset: Int): Unit =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support vMul operation")

    def vDiv(n: Int, a: Array[T], aOffset: Int, b: Array[T], bOffset: Int, y: Array[T],
      yOffset: Int): Unit =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support vDiv operation")

    def sum(n: Int, a: Array[T], aOffset: Int, stride: Int): T =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support sum operation")

    def arraycopy(src: Array[T], srcPos: Int,
      dest: Array[T], destPos: Int, length: Int): Unit =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support arraycopy operation")

    def getType(): TensorDataType =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support getType operation")

    def addcmul(value: T, n: Int,
      self: Array[T], selfOffset: Int,
      a: Array[T], aOffset: Int,
      b: Array[T], bOffset: Int): Unit =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support addcmul operation")

    def addcdiv(value: T, n: Int,
      self: Array[T], selfOffset: Int,
      a: Array[T], aOffset: Int,
      b: Array[T], bOffset: Int): Unit =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support addcdiv operation")

    def nearlyEqual(a: T, b: T, epsilon: Double): Boolean =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support nearlyEqual operation")

    override def floor(a: T): T =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support floor operation")

    override def ceil(a: T): T =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support ceil operation")

    override def isInf(a: T): Boolean =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support isInf operation")

    override def isFinite(a: T): Boolean =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support isFinite operation")

    override def isNan(a: T): Boolean =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support isNan operation")

    override def round(a: T): T =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support round operation")

    override def truncate(a: T): T =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support truncate operation")

    override def floorDiv(a: T, b: T): T =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support floorDiv operation")

    def clip(a: T, lower: T, upper: T): T =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support clip operation")

    def erf(x: T): T =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support erf operation")

    def erfc(x: T): T =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support erf operation")

    def logGamma(v: T): T =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support erf operation")

    def digamma(v: T): T =
      throw new UnsupportedOperationException(typeName +
        " in tensor does not support erf operation")
  }

  /**
   * Numerical operation for type T
   */
  class TensorNumericOps[@specialized(Float, Double) T](lhs: T)(implicit ev: TensorNumeric[T]) {
    // scalastyle:off methodName
    def +(rhs: T): T = ev.plus(lhs, rhs)

    def -(rhs: T): T = ev.minus(lhs, rhs)

    def *(rhs: T): T = ev.times(lhs, rhs)

    def /(rhs: T): T = ev.divide(lhs, rhs)

    // scalastyle:on methodName
  }

  object TensorNumeric {

    implicit object NumericFloat extends UndefinedTensorNumeric[Float]("Float") {
      override def plus(x: Float, y: Float): Float = x + y

      override def minus(x: Float, y: Float): Float = x - y

      override def times(x: Float, y: Float): Float = x * y

      override def divide(x: Float, y: Float): Float = x / y

      override def exp(x: Float): Float = java.lang.Math.exp(x).toFloat

      override def log(x: Float): Float = java.lang.Math.log(x).toFloat

      override def max(x: Float, y: Float): Float = java.lang.Math.max(x, y)

      override def min(x: Float, y: Float): Float = java.lang.Math.min(x, y)

      override def sqrt(x: Float): Float = Math.sqrt(x.toDouble).toFloat

      override def tanh(x: Float): Float = Math.tanh(x.toDouble).toFloat

      override def abs(x: Float): Float = Math.abs(x)

      override def negative(x: Float): Float = -x

      override def pow(x: Float): Float = Math.pow(x, -1).toFloat

      override def pow(x: Float, y: Float): Float = Math.pow(x, y).toFloat

      override def log1p(x: Float): Float = Math.log1p(x).toFloat

      override def isGreater(x: Float, y: Float): Boolean = x > y

      override def isGreaterEq(x: Float, y: Float): Boolean = x >= y

      override def rand(): Float = RNG.uniform(0, 1).toFloat

      override def randn(): Float = RNG.normal(0, 1).toFloat

      override def gemm(transa: Char, transb: Char, m: Int, n: Int, k: Int, alpha: Float,
        a: Array[Float], aOffset: Int, lda: Int, b: Array[Float], bOffset: Int, ldb: Int,
        beta: Float, c: Array[Float], cOffset: Int, ldc: Int): Unit = {
        require(MKL.isMKLLoaded, "mkl isn't loaded")
        MKL.vsgemm(transa, transb, m, n, k, alpha, a, aOffset, lda, b, bOffset,
          ldb, beta, c, cOffset, ldc)
      }

      override def gemv(trans: Char, m: Int, n: Int, alpha: Float,
        a: Array[Float], aoffset: Int, lda: Int,
        x: Array[Float], xOffset: Int, incx: Int, beta: Float,
        y: Array[Float], yOffset: Int,
        incy: Int): Unit = {
        require(MKL.isMKLLoaded, "mkl isn't loaded")
        MKL.vsgemv(trans, m, n, alpha, a, aoffset, lda, x, xOffset,
          incx, beta, y, yOffset, incy)
      }

      override def axpy(n: Int, da: Float, dx: Array[Float], _dx_offset: Int,
        incx: Int, dy: Array[Float],
        _dy_offset: Int, incy: Int): Unit = {
        require(MKL.isMKLLoaded, "mkl isn't loaded")
        MKL.vsaxpy(n, da, dx, _dx_offset, incx, dy, _dy_offset, incy)      }

      override def dot(n: Int, dx: Array[Float], _dx_offset: Int, incx: Int, dy: Array[Float],
        _dy_offset: Int, incy: Int): Float = {
        require(MKL.isMKLLoaded, "mkl isn't loaded")
        MKL.vsdot(n, dx, _dx_offset, incx, dy, _dy_offset, incy)      }

      override def ger(m: Int, n: Int, alpha: Float, x: Array[Float], _x_offset: Int, incx: Int,
        y: Array[Float], _y_offset: Int,
        incy: Int, a: Array[Float], _a_offset: Int, lda: Int): Unit = {
        require(MKL.isMKLLoaded, "mkl isn't loaded")
        MKL.vsger(m, n, alpha, x, _x_offset, incx, y, _y_offset,
          incy, a, _a_offset, lda)
      }

      override def fill(data: Array[Float], fromIndex: Int, toIndex: Int, value: Float): Unit = {
        util.Arrays.fill(data, fromIndex, toIndex, value)
      }

      override def fromType[@specialized(Float, Double, Int) K](k: K)(
        implicit c: ConvertableFrom[K]): Float =
        c.toFloat(k)

      override def toType[@specialized(Float, Double, Int) K]
      (t: Float)(implicit c: ConvertableTo[K]): K = c.fromFloat(t)

      override def getType(): TensorDataType = FloatType

      override def vPowx(n: Int, a: Array[Float], aOffset: Int, b: Float, y: Array[Float],
        yOffset: Int): Unit = {
        require(MKL.isMKLLoaded, "mkl isn't loaded")
        MKL.vsPowx(n, a, aOffset, b, y, yOffset)
      }

      override def vLn(n: Int, a: Array[Float], aOffset: Int, y: Array[Float], yOffset: Int)
      : Unit = {
        require(MKL.isMKLLoaded, "mkl isn't loaded")
        MKL.vsLn(n, a, aOffset, y, yOffset)
      }

      override def vExp(n: Int, a: Array[Float], aOffset: Int, y: Array[Float], yOffset: Int)
      : Unit = {
        require(MKL.isMKLLoaded, "mkl isn't loaded")
        MKL.vsExp(n, a, aOffset, y, yOffset)
      }

      override def vSqrt(n: Int, a: Array[Float], aOffset: Int, y: Array[Float], yOffset: Int)
      : Unit = {
        require(MKL.isMKLLoaded, "mkl isn't loaded")
        MKL.vsSqrt(n, a, aOffset, y, yOffset)
      }

      override def vTanh(n: Int, a: Array[Float], aOffset: Int, y: Array[Float], yOffset: Int)
      : Unit = {
        require(MKL.isMKLLoaded, "mkl isn't loaded")
        MKL.vsTanh(n, a, aOffset, y, yOffset)
      }

      override def vAbs(n: Int, a: Array[Float], aOffset: Int, y: Array[Float], yOffset: Int)
      : Unit = {
        require(MKL.isMKLLoaded, "mkl isn't loaded")
        MKL.vsAbs(n, a, aOffset, y, yOffset)
      }

      override def vLog1p(n: Int, a: Array[Float], aOffset: Int, y: Array[Float], yOffset: Int)
      : Unit = {
        require(MKL.isMKLLoaded, "mkl isn't loaded")
        MKL.vsLog1p(n, a, aOffset, y, yOffset)
      }

      override def scal(n: Int, sa: Float, sx: Array[Float], offset: Int, incx: Int): Unit = {
        require(MKL.isMKLLoaded, "mkl isn't loaded")
        MKL.vsscal(n, sa, sx, offset, incx)
      }

      override def inv(v: Float): Float = 1 / v

      override def add(n: Int, a: Array[Float], offset: Int, v: Float, stride: Int): Unit = {
        var i = 0
        while (i < n) {
          a(offset + i * stride) += v
          i += 1
        }
      }

      override def sub(n: Int, a: Array[Float], offset: Int, v: Float, stride: Int): Unit = {
        var i = 0
        while (i < n) {
          a(offset + i * stride) -= v
          i += 1
        }
      }

      override def vAdd(n: Int, a: Array[Float], aOffset: Int, b: Array[Float], bOffset: Int,
        y: Array[Float], yOffset: Int): Unit = {
        require(MKL.isMKLLoaded, "mkl isn't loaded")
        MKL.vsAdd(n, a, aOffset, b, bOffset, y, yOffset)
      }

      override def vSub(n: Int, a: Array[Float], aOffset: Int, b: Array[Float], bOffset: Int,
        y: Array[Float], yOffset: Int): Unit = {
        require(MKL.isMKLLoaded, "mkl isn't loaded")
        MKL.vsSub(n, a, aOffset, b, bOffset, y, yOffset)
      }

      override def vMul(n: Int, a: Array[Float], aOffset: Int, b: Array[Float], bOffset: Int,
        y: Array[Float], yOffset: Int): Unit = {
        if (MKL.isMKLLoaded) {
          MKL.vsMul(n, a, aOffset, b, bOffset, y, yOffset)
        } else {
          var i = 0
          while (i < n) {
            y(yOffset + i) = a(aOffset + i) * b(bOffset + i)
            i += 1
          }
        }
      }

      override def vDiv(n: Int, a: Array[Float], aOffset: Int, b: Array[Float], bOffset: Int,
        y: Array[Float], yOffset: Int): Unit = {
        if (MKL.isMKLLoaded) {
          MKL.vsDiv(n, a, aOffset, b, bOffset, y, yOffset)
        } else {
          var i = 0
          while (i < n) {
            y(yOffset + i) = a(aOffset + i) / b(bOffset + i)
            i += 1
          }
        }
      }

      override def prod(n: Int, a: Array[Float], aOffset: Int, stride: Int): Float = {
        var i = 0
        var r = 1.0f
        while (i < n) {
          r *= a(aOffset + i * stride)
          i += 1
        }
        r
      }

      override def sum(n: Int, a: Array[Float], aOffset: Int, stride: Int): Float = {
        var i = 0
        var r = 0.0f
        while (i < n) {
          r += a(aOffset + i * stride)
          i += 1
        }
        r
      }

      override def arraycopy(
            src: Array[Float],
            srcPos: Int,
            dest: Array[Float],
            destPos: Int,
            length: Int): Unit = {
        System.arraycopy(src, srcPos, dest, destPos, length)
      }


      override def nearlyEqual(a: Float, b: Float, epsilon: Double): Boolean = {
        val absA = math.abs(a)
        val absB = math.abs(b)
        val diff = math.abs(a - b)

        val result = if (a == b) {
          true
        } else if (a == 0 || b == 0 || diff < java.lang.Float.MIN_NORMAL) {
          diff < (epsilon * java.lang.Float.MIN_NORMAL)
        } else {
          diff / (absA + absB) < epsilon
        }

        result
      }

      override def addcmul(value: Float, n: Int,
        self: Array[Float], selfOffset: Int,
        a: Array[Float], aOffset: Int,
        b: Array[Float], bOffset: Int): Unit = {
        val v = value.asInstanceOf[Float]
        var i = 0

        while (i < n) {
          self(i + selfOffset) += a(aOffset + i) * b(bOffset + i) * v
          i += 1
        }
      }

      override def addcdiv(value: Float, n: Int,
        self: Array[Float], selfOffset: Int,
        a: Array[Float], aOffset: Int,
        b: Array[Float], bOffset: Int): Unit = {
        val v = value.asInstanceOf[Float]
        var i = 0

        while (i < n) {
          self(i + selfOffset) += a(aOffset + i) / b(bOffset + i) * v
          i += 1
        }
      }

      override def floor(a: Float): Float = math.floor(a).toFloat

      override def ceil(a: Float): Float = math.ceil(a).toFloat

      override def isFinite(a: Float): Boolean = !java.lang.Float.isInfinite(a)

      override def isNan(a: Float): Boolean = java.lang.Float.isNaN(a)

      override def isInf(a: Float): Boolean = java.lang.Float.isInfinite(a)

      override def round(a: Float): Float = Math.round(a).toFloat

      override def truncate(a: Float): Float = {
        if (a >= 0) {
          Math.floor(a).toFloat
        } else if (a == Math.floor(a)) {
          a
        } else {
          Math.floor(a).toFloat + 1
        }
      }

      override def floorDiv(a: Float, b: Float): Float = {
        Math.floor(a / b).toFloat
      }

      override def clip(a: Float, lower: Float, upper: Float): Float = {
        require(lower <= upper, "lower bound must be less or equal than upper bound")
        math.min(math.max(a, lower), upper)
      }

      override def erf(a: Float): Float = org.apache.commons.math3.special.Erf.erf(a).toFloat

      override def erfc(a: Float): Float = org.apache.commons.math3.special.Erf.erfc(a).toFloat

      override def logGamma(a: Float): Float =
        org.apache.commons.math3.special.Gamma.logGamma(a).toFloat

      override def digamma(a: Float): Float =
        org.apache.commons.math3.special.Gamma.digamma(a).toFloat
    }

    implicit object NumericDouble extends UndefinedTensorNumeric[Double]("Double") {
      override def plus(x: Double, y: Double): Double = x + y

      override def minus(x: Double, y: Double): Double = x - y

      override def times(x: Double, y: Double): Double = x * y

      override def divide(x: Double, y: Double): Double = x / y

      override def exp(x: Double): Double = java.lang.Math.exp(x)

      override def log(x: Double): Double = java.lang.Math.log(x)

      override def max(x: Double, y: Double): Double = java.lang.Math.max(x, y)

      override def min(x: Double, y: Double): Double = java.lang.Math.min(x, y)

      override def sqrt(x: Double): Double = Math.sqrt(x)

      override def tanh(x: Double): Double = Math.tanh(x)

      override def abs(x: Double): Double = Math.abs(x)

      override def negative(x: Double): Double = -x

      override def pow(x: Double): Double = Math.pow(x, -1)

      override def pow(x: Double, y: Double): Double = Math.pow(x, y)

      override def log1p(x: Double): Double = Math.log1p(x)

      override def isGreater(x: Double, y: Double): Boolean = x > y

      override def isGreaterEq(x: Double, y: Double): Boolean = x >= y

      override def rand(): Double = RNG.uniform(0, 1)

      override def randn(): Double = RNG.normal(0, 1)

      override def gemm(transa: Char, transb: Char, m: Int, n: Int, k: Int, alpha: Double,
        a: Array[Double], aOffset: Int, lda: Int, b: Array[Double], bOffset: Int, ldb: Int,
        beta: Double, c: Array[Double], cOffset: Int, ldc: Int): Unit = {
        require(MKL.isMKLLoaded, "mkl isn't loaded")

        MKL.vdgemm(transa, transb, m, n, k, alpha, a, aOffset, lda, b,
          bOffset, ldb, beta, c, cOffset, ldc)
      }

      override def gemv(trans: Char, m: Int, n: Int, alpha: Double, a: Array[Double], aoffset: Int,
        lda: Int, x: Array[Double], xOffset: Int, incx: Int, beta: Double, y: Array[Double],
        yOffset: Int, incy: Int): Unit = {
        require(MKL.isMKLLoaded, "mkl isn't loaded")
        MKL.vdgemv(trans, m, n, alpha, a, aoffset, lda, x, xOffset,
          incx, beta, y, yOffset, incy)
      }

      override def axpy(n: Int, da: Double, dx: Array[Double], _dx_offset: Int, incx: Int,
        dy: Array[Double], _dy_offset: Int, incy: Int): Unit = {
        require(MKL.isMKLLoaded, "mkl isn't loaded")
        MKL.vdaxpy(n, da, dx, _dx_offset, incx, dy, _dy_offset, incy)
      }

      override def dot(n: Int, dx: Array[Double], _dx_offset: Int, incx: Int, dy: Array[Double],
        _dy_offset: Int, incy: Int): Double = {
        require(MKL.isMKLLoaded, "mkl isn't loaded")
        MKL.vddot(n, dx, _dx_offset, incx, dy, _dy_offset, incy)
      }

      override def ger(m: Int, n: Int, alpha: Double, x: Array[Double], _x_offset: Int, incx: Int,
        y: Array[Double], _y_offset: Int,
        incy: Int, a: Array[Double], _a_offset: Int, lda: Int): Unit = {
        require(MKL.isMKLLoaded, "mkl isn't loaded")
        MKL.vdger(m, n, alpha, x, _x_offset, incx, y, _y_offset,
          incy, a, _a_offset, lda)
      }

      override def fill(data: Array[Double], fromIndex: Int, toIndex: Int, value: Double): Unit = {
        util.Arrays.fill(data, fromIndex, toIndex, value)
      }

      override def fromType[@specialized(Float, Double, Int) K](k: K)(
        implicit c: ConvertableFrom[K]): Double =
        c.toDouble(k)

      override def toType[@specialized(Float, Double, Int) K](t: Double)
        (implicit c: ConvertableTo[K]): K = c.fromDouble(t)

      override def getType(): TensorDataType = DoubleType

      override def vPowx(n: Int, a: Array[Double], aOffset: Int, b: Double, y: Array[Double],
        yOffset: Int): Unit = {
        require(MKL.isMKLLoaded, "mkl isn't loaded")
        MKL.vdPowx(n, a, aOffset, b, y, yOffset)
      }

      override def vLn(n: Int, a: Array[Double], aOffset: Int, y: Array[Double],
                       yOffset: Int): Unit = {
        require(MKL.isMKLLoaded, "mkl isn't loaded")
        MKL.vdLn(n, a, aOffset, y, yOffset)
      }

      override def vExp(n: Int, a: Array[Double], aOffset: Int, y: Array[Double],
                        yOffset: Int): Unit = {
        require(MKL.isMKLLoaded, "mkl isn't loaded")
        MKL.vdExp(n, a, aOffset, y, yOffset)
      }

      override def vSqrt(n: Int, a: Array[Double], aOffset: Int, y: Array[Double],
                         yOffset: Int): Unit = {
        require(MKL.isMKLLoaded, "mkl isn't loaded")
        MKL.vdSqrt(n, a, aOffset, y, yOffset)
      }

      override def vTanh(n: Int, a: Array[Double], aOffset: Int, y: Array[Double], yOffset: Int)
      : Unit = {
        require(MKL.isMKLLoaded, "mkl isn't loaded")
        MKL.vdTanh(n, a, aOffset, y, yOffset)
      }

      override def vAbs(n: Int, a: Array[Double], aOffset: Int, y: Array[Double], yOffset: Int)
      : Unit = {
        require(MKL.isMKLLoaded, "mkl isn't loaded")
        MKL.vdAbs(n, a, aOffset, y, yOffset)
      }

      override def vLog1p(n: Int, a: Array[Double], aOffset: Int, y: Array[Double], yOffset: Int)
      : Unit = {
        require(MKL.isMKLLoaded, "mkl isn't loaded")
        MKL.vdLog1p(n, a, aOffset, y, yOffset)
      }

      override def scal(n: Int, sa: Double, sx: Array[Double], offset: Int, incx: Int): Unit = {
        require(MKL.isMKLLoaded, "mkl isn't loaded")
        MKL.vdscal(n, sa, sx, offset, incx)
      }

      override def inv(v: Double): Double = 1 / v

      override def add(n: Int, a: Array[Double], offset: Int, v: Double, stride: Int): Unit = {
        var i = 0
        while (i < n) {
          a(offset + i * stride) += v
          i += 1
        }
      }

      override def sub(n: Int, a: Array[Double], offset: Int, v: Double, stride: Int): Unit = {
        var i = 0
        while (i < n) {
          a(offset + i * stride) -= v
          i += 1
        }
      }

      override def vAdd(n: Int, a: Array[Double], aOffset: Int, b: Array[Double], bOffset: Int,
        y: Array[Double], yOffset: Int): Unit = {
        require(MKL.isMKLLoaded, "mkl isn't loaded")
        MKL.vdAdd(n, a, aOffset, b, bOffset, y, yOffset)
      }

      override def vSub(n: Int, a: Array[Double], aOffset: Int, b: Array[Double], bOffset: Int,
        y: Array[Double], yOffset: Int): Unit = {
        require(MKL.isMKLLoaded, "mkl isn't loaded")
        MKL.vdSub(n, a, aOffset, b, bOffset, y, yOffset)
      }

      override def vMul(n: Int, a: Array[Double], aOffset: Int, b: Array[Double], bOffset: Int,
        y: Array[Double], yOffset: Int): Unit = {
        if (MKL.isMKLLoaded) {
          MKL.vdMul(n, a, aOffset, b, bOffset, y, yOffset)
        } else {
          var i = 0
          while (i < n) {
            y(yOffset + i) = a(aOffset + i) * b(bOffset + i)
            i += 1
          }
        }
      }

      override def vDiv(n: Int, a: Array[Double], aOffset: Int, b: Array[Double], bOffset: Int,
        y: Array[Double], yOffset: Int): Unit = {
        if (MKL.isMKLLoaded) {
          MKL.vdDiv(n, a, aOffset, b, bOffset, y, yOffset)
        } else {
          var i = 0
          while (i < n) {
            y(yOffset + i) = a(aOffset + i) / b(bOffset + i)
            i += 1
          }
        }
      }

      override def prod(n: Int, a: Array[Double], aOffset: Int, stride: Int): Double = {
        var i = 0
        var r = 1.0
        while (i < n) {
          r *= a(aOffset + i * stride)
          i += 1
        }
        r
      }

      override def sum(n: Int, a: Array[Double], aOffset: Int, stride: Int): Double = {
        var i = 0
        var r = 0.0
        while (i < n) {
          r += a(aOffset + i * stride)
          i += 1
        }
        r
      }

      override def arraycopy(
            src: Array[Double],
            srcPos: Int,
            dest: Array[Double],
            destPos: Int,
            length: Int): Unit = {
        System.arraycopy(src, srcPos, dest, destPos, length)
      }

      override def nearlyEqual(a: Double, b: Double, epsilon: Double): Boolean = {
        val absA = math.abs(a)
        val absB = math.abs(b)
        val diff = math.abs(a - b)

        val result = if (a == b) {
          true
        } else if (a == 0 || b == 0 || diff < java.lang.Double.MIN_NORMAL) {
          diff < (epsilon * java.lang.Double.MIN_NORMAL)
        } else {
          diff / (absA + absB) < epsilon
        }

        if (!result) {
          if (a == b) {
            true
          } else if (a == 0 || b == 0 || diff < java.lang.Double.MIN_NORMAL) {
            diff < (epsilon * java.lang.Double.MIN_NORMAL)
          } else {
            diff / (absA + absB) < epsilon
          }
        }
        result
      }

      override def addcmul(value: Double, n: Int,
        self: Array[Double], selfOffset: Int,
        a: Array[Double], aOffset: Int,
        b: Array[Double], bOffset: Int): Unit = {
        val v = value.asInstanceOf[Double]
        var i = 0

        while (i < n) {
          self(i + selfOffset) += a(aOffset + i) * b(bOffset + i) * v
          i += 1
        }
      }

      override def addcdiv(value: Double, n: Int,
        self: Array[Double], selfOffset: Int,
        a: Array[Double], aOffset: Int,
        b: Array[Double], bOffset: Int): Unit = {
        val v = value.asInstanceOf[Double]
        var i = 0

        while (i < n) {
          self(i + selfOffset) += a(aOffset + i) / b(bOffset + i) * v
          i += 1
        }
      }

      override def floor(a: Double): Double = math.floor(a)

      override def ceil(a: Double): Double = math.ceil(a)

      override def isFinite(a: Double): Boolean = !java.lang.Double.isInfinite(a)

      override def isNan(a: Double): Boolean = java.lang.Double.isNaN(a)

      override def isInf(a: Double): Boolean = java.lang.Double.isInfinite(a)

      override def round(a: Double): Double = Math.round(a).toDouble

      override def truncate(a: Double): Double = {
        if (a >= 0) {
          Math.floor(a)
        } else if (a == Math.floor(a)) {
          a
        } else {
          Math.floor(a) + 1
        }
      }

      override def floorDiv(a: Double, b: Double): Double = {
        Math.floor(a / b)
      }

      override def clip(a: Double, lower: Double, upper: Double): Double = {
        require(lower <= upper, "lower bound must be less or equal than upper bound")
        math.min(math.max(a, lower), upper)
      }

      override def erf(a: Double): Double = org.apache.commons.math3.special.Erf.erf(a)

      override def erfc(a: Double): Double = org.apache.commons.math3.special.Erf.erfc(a)

      override def logGamma(a: Double): Double =
        org.apache.commons.math3.special.Gamma.logGamma(a)

      override def digamma(a: Double): Double =
        org.apache.commons.math3.special.Gamma.digamma(a)
    }

    implicit object NumericString extends UndefinedTensorNumeric[String]("String") {
      override def plus(x: String, y: String): String = x + y

      override def getType(): TensorDataType = StringType

      override def fromType[K](k: K)(
        implicit c: ConvertableFrom[K]): String =
        c.toString(k)

      override def axpy(n: Int, da: String, dx: Array[String], _dx_offset: Int,
        incx: Int, dy: Array[String],
        _dy_offset: Int, incy: Int): Unit = {
        var i = 0
        while (i < n) {
          dy(i + _dy_offset) = dx(_dx_offset + i) + dy(_dy_offset + i)
          i += 1
        }
      }

      override def nearlyEqual(a: String, b: String, epsilon: Double): Boolean = {
        a == b
      }
    }

    implicit object NumericBoolean extends UndefinedTensorNumeric[Boolean]("Boolean") {
      override def getType(): TensorDataType = BooleanType

      override def or(x: Boolean, y: Boolean): Boolean = x || y

      override def and(x: Boolean, y: Boolean): Boolean = x && y

      override def fromType[K](k: K)(
        implicit c: ConvertableFrom[K]): Boolean =
        c.toBoolean(k)

      override def toType[K](t: Boolean)(
        implicit c: ConvertableTo[K]): K = c.fromBoolean(t)

      override def nearlyEqual(a: Boolean, b: Boolean, epsilon: Double): Boolean = {
        a == b
      }
    }

    implicit object NumericInt extends UndefinedTensorNumeric[Int]("Int") {
      override def getType(): TensorDataType = IntType

      override def plus(x: Int, y: Int): Int = x + y

      override def minus(x: Int, y: Int): Int = x - y

      override def times(x: Int, y: Int): Int = x * y

      override def divide(x: Int, y: Int): Int = x / y

      override def exp(x: Int): Int = java.lang.Math.exp(x).toInt

      override def log(x: Int): Int = java.lang.Math.log(x).toInt

      override def max(x: Int, y: Int): Int = java.lang.Math.max(x, y)

      override def min(x: Int, y: Int): Int = java.lang.Math.min(x, y)

      override def sqrt(x: Int): Int = Math.sqrt(x.toDouble).toInt

      override def tanh(x: Int): Int = Math.tanh(x.toDouble).toInt

      override def fromType[K](k: K)(
        implicit c: ConvertableFrom[K]): Int =
        c.toInt(k)

      override def toType[K](t: Int)
        (implicit c: ConvertableTo[K]): K = c.fromInt(t)

      override def axpy(n: Int, da: Int, dx: Array[Int], _dx_offset: Int,
        incx: Int, dy: Array[Int],
        _dy_offset: Int, incy: Int): Unit = {
        var i = 0
        while (i < n) {
          dy(i + _dy_offset) = dx(_dx_offset + i) + dy(_dy_offset + i)
          i += 1
        }
      }

      override def abs(x: Int): Int = Math.abs(x)

      override def negative(x: Int): Int = -x

      override def pow(x: Int): Int = Math.pow(x, -1).toInt

      override def pow(x: Int, y: Int): Int = Math.pow(x, y).toInt

      override def log1p(x: Int): Int = Math.log1p(x).toInt

      override def isGreater(x: Int, y: Int): Boolean = x > y

      override def isGreaterEq(x: Int, y: Int): Boolean = x >= y

      override def nearlyEqual(a: Int, b: Int, epsilon: Double): Boolean = a == b

      override def prod(n: Int, a: Array[Int], aOffset: Int, stride: Int): Int = {
        var i = 0
        var r = 1
        while (i < n) {
          r *= a(aOffset + i * stride)
          i += 1
        }
        r
      }

      override def sum(n: Int, a: Array[Int], aOffset: Int, stride: Int): Int = {
        var i = 0
        var r = 0
        while (i < n) {
          r += a(aOffset + i * stride)
          i += 1
        }
        r
      }

      override def floor(a: Int): Int = a

      override def sub(n: Int, a: Array[Int], offset: Int, v: Int, stride: Int): Unit = {
        var i = 0
        while(i < n) {
          a(i * stride + offset) -= v
          i += 1
        }
      }

      override def round(a: Int): Int = a

      override def vDiv(n: Int, a: Array[Int], aOffset: Int, b: Array[Int], bOffset: Int,
        y: Array[Int], yOffset: Int): Unit = {
        var i = 0
        while(i < n) {
          y(i + yOffset) = a(i + aOffset) / b(i + bOffset)
          i += 1
        }
      }

      override def vMul(n: Int, a: Array[Int], aOffset: Int, b: Array[Int], bOffset: Int,
        y: Array[Int], yOffset: Int): Unit = {
        var i = 0
        while(i < n) {
          y(i + yOffset) = a(i + aOffset) * b(i + bOffset)
          i += 1
        }
      }

      override def truncate(a: Int): Int = a

      override def floorDiv(a: Int, b: Int): Int = {
        var var2 = a / b
        if ((a ^ b) < 0 && var2 * b != a) {
          var2 -= 1
        }
        var2
      }
    }

    implicit object NumericLong extends UndefinedTensorNumeric[Long]("Long") {
      override def getType(): TensorDataType = LongType

      override def plus(x: Long, y: Long): Long = x + y

      override def minus(x: Long, y: Long): Long = x - y

      override def times(x: Long, y: Long): Long = x * y

      override def divide(x: Long, y: Long): Long = x / y

      override def exp(x: Long): Long = java.lang.Math.exp(x).toLong

      override def log(x: Long): Long = java.lang.Math.log(x).toLong

      override def max(x: Long, y: Long): Long = java.lang.Math.max(x, y)

      override def min(x: Long, y: Long): Long = java.lang.Math.min(x, y)

      override def sqrt(x: Long): Long = Math.sqrt(x.toDouble).toLong

      override def tanh(x: Long): Long = Math.tanh(x.toDouble).toLong

      override def abs(x: Long): Long = Math.abs(x)

      override def negative(x: Long): Long = -x

      override def pow(x: Long): Long = Math.pow(x, -1).toLong

      override def pow(x: Long, y: Long): Long = Math.pow(x, y).toLong

      override def log1p(x: Long): Long = Math.log1p(x).toLong

      override def isGreater(x: Long, y: Long): Boolean = x > y

      override def isGreaterEq(x: Long, y: Long): Boolean = x >= y

      override def fromType[K](k: K)(
        implicit c: ConvertableFrom[K]): Long =
        c.toLong(k)

      override def toType[@specialized(Float, Double, Int) K](t: Long)
        (implicit c: ConvertableTo[K]): K = c.fromLong(t)

      override def axpy(n: Int, da: Long, dx: Array[Long], _dx_offset: Int,
        incx: Int, dy: Array[Long],
        _dy_offset: Int, incy: Int): Unit = {
        var i = 0
        while (i < n) {
          dy(i + _dy_offset) = dx(_dx_offset + i) + dy(_dy_offset + i)
          i += 1
        }
      }

      override def nearlyEqual(a: Long, b: Long, epsilon: Double): Boolean = {
        val absA = math.abs(a)
        val absB = math.abs(b)
        val diff = math.abs(a - b)

        val result = if (a == b) {
          true
        } else if (a == 0 || b == 0 || diff < java.lang.Float.MIN_NORMAL) {
          diff < (epsilon * java.lang.Float.MIN_NORMAL)
        } else {
          diff / (absA + absB) < epsilon
        }

        result
      }

      override def floor(a: Long): Long = a
    }

    implicit object NumericShort extends UndefinedTensorNumeric[Short]("Short") {
      override def getType(): TensorDataType = ShortType

      override def plus(x: Short, y: Short): Short = (x + y).toShort

      override def minus(x: Short, y: Short): Short = (x - y).toShort

      override def times(x: Short, y: Short): Short = (x * y).toShort

      override def divide(x: Short, y: Short): Short = (x / y).toShort

      override def exp(x: Short): Short = java.lang.Math.exp(x).toShort

      override def log(x: Short): Short = java.lang.Math.log(x).toShort

      override def max(x: Short, y: Short): Short = java.lang.Math.max(x, y).toShort

      override def min(x: Short, y: Short): Short = java.lang.Math.min(x, y).toShort

      override def sqrt(x: Short): Short = Math.sqrt(x.toDouble).toShort

      override def tanh(x: Short): Short = Math.tanh(x.toDouble).toShort

      override def abs(x: Short): Short = Math.abs(x).toShort

      override def negative(x: Short): Short = (-x).toShort

      override def pow(x: Short): Short = Math.pow(x, -1).toShort

      override def pow(x: Short, y: Short): Short = Math.pow(x, y).toShort

      override def log1p(x: Short): Short = Math.log1p(x).toShort

      override def isGreater(x: Short, y: Short): Boolean = x > y

      override def isGreaterEq(x: Short, y: Short): Boolean = x >= y

      override def fromType[K](k: K)(
        implicit c: ConvertableFrom[K]): Short =
        c.toShort(k)

      override def toType[@specialized(Float, Double, Int) K](t: Short)
        (implicit c: ConvertableTo[K]): K = c.fromShort(t)

      override def axpy(n: Int, da: Short, dx: Array[Short], _dx_offset: Int,
        incx: Int, dy: Array[Short],
        _dy_offset: Int, incy: Int): Unit = {
        var i = 0
        while (i < n) {
          dy(i + _dy_offset) = (dx(_dx_offset + i) + dy(_dy_offset + i)).toShort
          i += 1
        }
      }

      override def nearlyEqual(a: Short, b: Short, epsilon: Double): Boolean = {
        val absA = math.abs(a)
        val absB = math.abs(b)
        val diff = math.abs(a - b)

        val result = if (a == b) {
          true
        } else if (a == 0 || b == 0 || diff < java.lang.Float.MIN_NORMAL) {
          diff < (epsilon * java.lang.Float.MIN_NORMAL)
        } else {
          diff / (absA + absB) < epsilon
        }

        result
      }

      override def floor(a: Short): Short = a
    }

    implicit object NumericChar extends UndefinedTensorNumeric[Char]("Char") {
      override def getType(): TensorDataType = CharType

      override def plus(x: Char, y: Char): Char = (x + y).toChar

      override def minus(x: Char, y: Char): Char = (x - y).toChar

      override def fromType[K](k: K)(
        implicit c: ConvertableFrom[K]): Char =
        c.toChar(k)

      override def axpy(n: Int, da: Char, dx: Array[Char], _dx_offset: Int,
        incx: Int, dy: Array[Char],
        _dy_offset: Int, incy: Int): Unit = {
        var i = 0
        while (i < n) {
          dy(i + _dy_offset) = (dx(_dx_offset + i) + dy(_dy_offset + i)).toChar
          i += 1
        }
      }

      override def nearlyEqual(a: Char, b: Char, epsilon: Double): Boolean = {
        a == b
      }
    }

    implicit object NumericByte extends UndefinedTensorNumeric[Byte]("Byte") {
      override def getType(): TensorDataType = ByteType

      override def plus(x: Byte, y: Byte): Byte = (x + y).toByte

      override def minus(x: Byte, y: Byte): Byte = (x - y).toByte

      override def fromType[K](k: K)(
        implicit c: ConvertableFrom[K]): Byte =
        c.toByte(k)

      override def axpy(n: Int, da: Byte, dx: Array[Byte], _dx_offset: Int,
        incx: Int, dy: Array[Byte],
        _dy_offset: Int, incy: Int): Unit = {
        var i = 0
        while (i < n) {
          dy(i + _dy_offset) = (dx(_dx_offset + i) + dy(_dy_offset + i)).toByte
          i += 1
        }
      }

      override def nearlyEqual(a: Byte, b: Byte, epsilon: Double): Boolean = {
        a == b
      }
    }
  }
}
