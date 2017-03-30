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

class TensorNumericMath

object TensorNumericMath {

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

    def sqrt(x: T): T

    def abs(x: T): T

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

    def fromType[@specialized(Float, Double, Int) K](k: K)(implicit c: ConvertableFrom[K]): T

    def toType[@specialized(Float, Double, Int) K](t: T)(implicit c: ConvertableTo[K]): K

    def vPowx(n: Int, a: Array[T], aOffset: Int, b: T, y: Array[T], yOffset: Int): Unit

    def vLn(n: Int, a: Array[T], aOffset: Int, y: Array[T], yOffset: Int): Unit

    def vExp(n: Int, a: Array[T], aOffset: Int, y: Array[T], yOffset: Int): Unit

    def vSqrt(n: Int, a: Array[T], aOffset: Int, y: Array[T], yOffset: Int): Unit

    def vAbs(n: Int, a: Array[T], aOffset: Int, y: Array[T], yOffset: Int): Unit

    def vLog1p(n: Int, a: Array[T], aOffset: Int, y: Array[T], yOffset: Int): Unit

    def scal(n: Int, sa: T, sx: Array[T], offset: Int, incx: Int): Unit

    def inv(v: T): T

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

    def getType(): TensorDataType
  }

  class TensorNumericOps[@specialized(Float, Double) T](lhs: T)(implicit ev: TensorNumeric[T]) {
    // scalastyle:off methodName
    def +(rhs: T): T = ev.plus(lhs, rhs)

    def -(rhs: T): T = ev.minus(lhs, rhs)

    def *(rhs: T): T = ev.times(lhs, rhs)

    def /(rhs: T): T = ev.divide(lhs, rhs)

    // scalastyle:on methodName
  }

  object TensorNumeric {

    implicit object NumericFloat extends TensorNumeric[Float] {
      def plus(x: Float, y: Float): Float = x + y

      def minus(x: Float, y: Float): Float = x - y

      def times(x: Float, y: Float): Float = x * y

      def divide(x: Float, y: Float): Float = x / y

      def exp(x: Float): Float = java.lang.Math.exp(x).toFloat

      def log(x: Float): Float = java.lang.Math.log(x).toFloat

      def max(x: Float, y: Float): Float = java.lang.Math.max(x, y)

      def sqrt(x: Float): Float = Math.sqrt(x.toDouble).toFloat

      def abs(x: Float): Float = Math.abs(x)

      def negative(x: Float): Float = -x

      def pow(x: Float): Float = Math.pow(x, -1).toFloat

      def pow(x: Float, y: Float): Float = Math.pow(x, y).toFloat

      def log1p(x: Float): Float = Math.log1p(x).toFloat

      def isGreater(x: Float, y: Float): Boolean = (x > y)

      def isGreaterEq(x: Float, y: Float): Boolean = (x >= y)

      def rand(): Float = RNG.uniform(0, 1).toFloat

      def randn(): Float = RNG.normal(0, 1).toFloat

      def gemm(transa: Char, transb: Char, m: Int, n: Int, k: Int, alpha: Float,
        a: Array[Float], aOffset: Int, lda: Int, b: Array[Float], bOffset: Int, ldb: Int,
        beta: Float, c: Array[Float], cOffset: Int, ldc: Int): Unit = {
        require(MKL.isMKLLoaded, "mkl isn't loaded")
        MKL.vsgemm(transa, transb, m, n, k, alpha, a, aOffset, lda, b, bOffset,
          ldb, beta, c, cOffset, ldc)
      }

      def gemv(trans: Char, m: Int, n: Int, alpha: Float, a: Array[Float], aoffset: Int, lda: Int,
        x: Array[Float], xOffset: Int, incx: Int, beta: Float, y: Array[Float], yOffset: Int,
        incy: Int): Unit = {
        require(MKL.isMKLLoaded, "mkl isn't loaded")
        MKL.vsgemv(trans, m, n, alpha, a, aoffset, lda, x, xOffset,
          incx, beta, y, yOffset, incy)
      }

      def axpy(n: Int, da: Float, dx: Array[Float], _dx_offset: Int, incx: Int, dy: Array[Float],
        _dy_offset: Int, incy: Int): Unit = {
        require(MKL.isMKLLoaded, "mkl isn't loaded")
        MKL.vsaxpy(n, da, dx, _dx_offset, incx, dy, _dy_offset, incy)      }

      def dot(n: Int, dx: Array[Float], _dx_offset: Int, incx: Int, dy: Array[Float],
        _dy_offset: Int, incy: Int): Float = {
        require(MKL.isMKLLoaded, "mkl isn't loaded")
        MKL.vsdot(n, dx, _dx_offset, incx, dy, _dy_offset, incy)      }

      def ger(m: Int, n: Int, alpha: Float, x: Array[Float], _x_offset: Int, incx: Int,
        y: Array[Float], _y_offset: Int,
        incy: Int, a: Array[Float], _a_offset: Int, lda: Int): Unit = {
        require(MKL.isMKLLoaded, "mkl isn't loaded")
        MKL.vsger(m, n, alpha, x, _x_offset, incx, y, _y_offset,
          incy, a, _a_offset, lda)
      }

      def fill(data: Array[Float], fromIndex: Int, toIndex: Int, value: Float): Unit = {
        util.Arrays.fill(data, fromIndex, toIndex, value)
      }

      def fromType[@specialized(Float, Double, Int) K](k: K)(
        implicit c: ConvertableFrom[K]): Float =
        c.toFloat(k)

      def toType[@specialized(Float, Double, Int) K](t: Float)(implicit c: ConvertableTo[K]): K =
        c.fromFloat(t)

      def getType(): TensorDataType = FloatType

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

      override def sum(n: Int, a: Array[Float], aOffset: Int, stride: Int): Float = {
        var i = 0
        var r = 0.0f
        while (i < n) {
          r += a(aOffset + i * stride)
          i += 1
        }
        r
      }
    }

    implicit object NumericDouble extends TensorNumeric[Double] {
      def plus(x: Double, y: Double): Double = x + y

      def minus(x: Double, y: Double): Double = x - y

      def times(x: Double, y: Double): Double = x * y

      def divide(x: Double, y: Double): Double = x / y

      def exp(x: Double): Double = java.lang.Math.exp(x)

      def log(x: Double): Double = java.lang.Math.log(x)

      def max(x: Double, y: Double): Double = java.lang.Math.max(x, y)

      def sqrt(x: Double): Double = Math.sqrt(x)

      def abs(x: Double): Double = Math.abs(x)

      def negative(x: Double): Double = -x

      def pow(x: Double): Double = Math.pow(x, -1)

      def pow(x: Double, y: Double): Double = Math.pow(x, y)

      def log1p(x: Double): Double = Math.log1p(x)

      def isGreater(x: Double, y: Double): Boolean = (x > y)

      def isGreaterEq(x: Double, y: Double): Boolean = (x >= y)

      def rand(): Double = RNG.uniform(0, 1)

      def randn(): Double = RNG.normal(0, 1)

      def gemm(transa: Char, transb: Char, m: Int, n: Int, k: Int, alpha: Double,
        a: Array[Double], aOffset: Int, lda: Int, b: Array[Double], bOffset: Int, ldb: Int,
        beta: Double, c: Array[Double], cOffset: Int, ldc: Int): Unit = {
        require(MKL.isMKLLoaded, "mkl isn't loaded")

        MKL.vdgemm(transa, transb, m, n, k, alpha, a, aOffset, lda, b,
          bOffset, ldb, beta, c, cOffset, ldc)
      }

      def gemv(trans: Char, m: Int, n: Int, alpha: Double, a: Array[Double], aoffset: Int,
        lda: Int, x: Array[Double], xOffset: Int, incx: Int, beta: Double, y: Array[Double],
        yOffset: Int, incy: Int): Unit = {
        require(MKL.isMKLLoaded, "mkl isn't loaded")
        MKL.vdgemv(trans, m, n, alpha, a, aoffset, lda, x, xOffset,
          incx, beta, y, yOffset, incy)
      }

      def axpy(n: Int, da: Double, dx: Array[Double], _dx_offset: Int, incx: Int,
        dy: Array[Double], _dy_offset: Int, incy: Int): Unit = {
        require(MKL.isMKLLoaded, "mkl isn't loaded")
        MKL.vdaxpy(n, da, dx, _dx_offset, incx, dy, _dy_offset, incy)
      }

      def dot(n: Int, dx: Array[Double], _dx_offset: Int, incx: Int, dy: Array[Double],
        _dy_offset: Int, incy: Int): Double = {
        require(MKL.isMKLLoaded, "mkl isn't loaded")
        MKL.vddot(n, dx, _dx_offset, incx, dy, _dy_offset, incy)
      }

      def ger(m: Int, n: Int, alpha: Double, x: Array[Double], _x_offset: Int, incx: Int,
        y: Array[Double], _y_offset: Int,
        incy: Int, a: Array[Double], _a_offset: Int, lda: Int): Unit = {
        require(MKL.isMKLLoaded, "mkl isn't loaded")
        MKL.vdger(m, n, alpha, x, _x_offset, incx, y, _y_offset,
          incy, a, _a_offset, lda)
      }

      def fill(data: Array[Double], fromIndex: Int, toIndex: Int, value: Double): Unit = {
        util.Arrays.fill(data, fromIndex, toIndex, value)
      }

      def fromType[@specialized(Float, Double, Int) K](k: K)(
        implicit c: ConvertableFrom[K]): Double =
        c.toDouble(k)

      def toType[@specialized(Float, Double, Int) K](t: Double)(implicit c: ConvertableTo[K]): K =
        c.fromDouble(t)

      def getType(): TensorDataType = DoubleType

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

      override def sum(n: Int, a: Array[Double], aOffset: Int, stride: Int): Double = {
        var i = 0
        var r = 0.0
        while (i < n) {
          r += a(aOffset + i * stride)
          i += 1
        }
        r
      }
    }
  }
}
