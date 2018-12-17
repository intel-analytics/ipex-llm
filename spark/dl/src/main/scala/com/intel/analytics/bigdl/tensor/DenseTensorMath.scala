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

import com.intel.analytics.bigdl.mkl.MKL
import com.intel.analytics.bigdl.tensor.TensorNumericMath._
import com.intel.analytics.bigdl.tensor.{DenseTensorApply => Apply}

import scala.reflect.ClassTag

object DenseTensorMath {
  val taskSize: Int = System.getProperty("cpu.task.size", "250000").toInt

  def mul[@specialized(Float, Double) T](self: DenseTensor[T], x: Tensor[T], value: T)
    (implicit ev: TensorNumeric[T]): Tensor[T] = {
    if (x != null) {
      require(self.nElement() == x.nElement())
      self.copy(x)
    }

    if (self.isContiguous()) {
      ev.scal(self.nElement, value, self.storage().array(), self.storageOffset() - 1, 1)
    } else {
      val func = new TensorFunc2[T] {
        override def apply(data: Array[T], index: Int): Unit = {
          data(index) = ev.times(data(index), value)
        }
      }
      Apply.apply1[T](self, func)
    }
    self
  }

  def cmul[@specialized T](self: DenseTensor[T], x: DenseTensor[T], y: DenseTensor[T])
    (implicit ev: TensorNumeric[T]): Tensor[T] = {
    if (x.nElement() != y.nElement() && DenseTensor.canFastBroadcast(x, y)) {
      require(self.nElement() == x.nElement(), "the self tensor nElement is not same as x" +
        s"self(${self.nElement()}) x(${x.nElement()})")
      // recursive cmul
      var i = 0
      while(i < x.size(1)) {
        cmul(self.select(1, i + 1).asInstanceOf[DenseTensor[T]],
          x.select(1, i + 1).asInstanceOf[DenseTensor[T]], y)
        i += 1
      }
    } else if (x.nElement() != y.nElement() && DenseTensor.canFastBroadcast(y, x)) {
      require(self.nElement() == y.nElement(), "the self tensor nElement is not same as y" +
        s"self(${self.nElement()}) y(${y.nElement()})")
      // recursive cmul
      var i = 0
      while(i < y.size(1)) {
        cmul(self.select(1, i + 1).asInstanceOf[DenseTensor[T]], x,
          y.select(1, i + 1).asInstanceOf[DenseTensor[T]])
        i += 1
      }
    } else if (x.nElement() != y.nElement()) {
      self.resizeAs(x).copy(x)
      self.cmul(self.expandTensor(y))
    } else {
      require(self.nElement() == y.nElement(), s"element number doesn't match " +
        s"self(${self.nElement()}) y(${y.nElement()}) x(${x.nElement()})")
      if (self.isContiguous() && x.isContiguous() && y.isContiguous() && MKL.isMKLLoaded) {

        ev.vMul(self.nElement(), x.storage().array(), x.storageOffset() - 1,
          y.storage().array(), y.storageOffset() - 1, self.storage().array(), self.storageOffset()
            - 1)
      } else {
        val func6 = new TensorFunc6[T] {
          override def apply(data1: Array[T], offset1: Int, data2: Array[T], offset2: Int,
            data3: Array[T], offset3: Int): Unit = {
            data1(offset1) = ev.times(data2(offset2), data3(offset3))
          }
        }
        val func4 = new TensorFunc4[T] {
          override def apply(data1: Array[T], offset1: Int, data2: Array[T], offset2: Int): Unit = {
            data1(offset1) = ev.times(data1(offset1), data2(offset2))
          }
        }
        // For special case, we can use apply2 to instead of apply3
        if (self == y) {
          Apply.apply2(self, x, func4)
        } else if (self == x) {
          Apply.apply2(self, y, func4)
        } else {
          Apply.apply3[T](self, x, y, func6)
        }
      }
    }
    self
  }

  def cdiv[@specialized(Float, Double) T](self: DenseTensor[T], x: Tensor[T], y: Tensor[T])
    (implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(self.nElement() == y.nElement() && self.nElement() == x.nElement(),
      "element number doesn't match")
    if (self.isContiguous() && y.isContiguous() && x.isContiguous() && MKL.isMKLLoaded) {

      ev.vDiv(self.nElement(), x.storage().array(), x.storageOffset() - 1,
        y.storage().array(), y.storageOffset() - 1, self.storage().array(), self.storageOffset()
          - 1)
    } else {
      val func = new TensorFunc6[T] {
        override def apply(data1: Array[T], offset1: Int, data2: Array[T], offset2: Int,
                          data3: Array[T], offset3: Int): Unit = {
          data1(offset1) = ev.divide(data2(offset2), data3(offset3))
        }
      }
      Apply.apply3[T](self, x, y, func)
    }
    self
  }

  def cadd[@specialized(Float, Double) T](
    self: DenseTensor[T], x: Tensor[T], value: T, y: Tensor[T])
    (implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(x != null && y.nElement() == x.nElement())

    if (!self.eq(x) && !self.eq(y)) {
      self.resizeAs(x).copy(x)
    }

    if (self.eq(x) && self.isContiguous() && y.isContiguous()) {
      ev.axpy(y.nElement(), value, y.storage().array(), y.storageOffset() - 1, 1,
        self.storage().array(), self.storageOffset() - 1, 1)
    } else {
      val func = new TensorFunc6[T] {
        override def apply(data1: Array[T], offset1: Int, data2: Array[T], offset2: Int,
                          data3: Array[T], offset3: Int): Unit = {
          data1(offset1) = ev.plus(data2(offset2), ev.times(value, data3(offset3)))
        }
      }
      Apply.apply3[T](self, x, y, func)
    }
    self
  }

  def csub[@specialized(Float, Double) T]
  (self: DenseTensor[T], x: Tensor[T], value: T, y: Tensor[T])
  (implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(x != null && x.nElement() == y.nElement())
    if(!self.eq(x)) {
      self.resizeAs(x).copy(x)
    }

    if(self.eq(x) && self.isContiguous() && y.isContiguous()) {
      ev.axpy(y.nElement(), value, y.storage().array(),
        y.storageOffset() - 1, 1, self.storage().array(), self.storageOffset() - 1, 1)
    } else {
      val func2 = new TensorFunc4[T] {
        override def apply(data1: Array[T], offset1: Int, data2: Array[T], offset2: Int): Unit =
        { data1(offset1) = ev.minus(data1(offset1), ev.times(value, data2(offset2)))  }}
      Apply.apply2[T](self, y, func2)
    }
    self
  }

  def add[@specialized(Float, Double) T: ClassTag](s: T, t: DenseTensor[T])
    (implicit ev: TensorNumeric[T]): Tensor[T] = {
    val result = new DenseTensor[T]()
    result.resizeAs(t)
    result.copy(t)
    val func = new TensorFunc2[T] {
      override def apply(data: Array[T], index: Int): Unit = {
        data(index) = ev.plus(data(index), s)
      }
    }
    Apply.apply1[T](result, func)

    result
  }

  def add[@specialized(Float, Double) T: ClassTag](self: DenseTensor[T], t: Tensor[T])
    (implicit ev: TensorNumeric[T]): Tensor[T] = {
    val result = new DenseTensor[T]()
    result.resizeAs(self)
    result.copy(self)
    val n = result.nElement()
    if (result.isContiguous() && t.isContiguous() && n == t.nElement()) {
      ev.axpy(n, ev.fromType[Int](1), t.storage().array(), t.storageOffset() - 1, 1,
        result.storage().array,
        result.storageOffset() - 1, 1)
      result
    } else {
      val func2 = new TensorFunc4[T] {
        override def apply(data1: Array[T], offset1: Int, data2: Array[T], offset2: Int): Unit = {
          data1(offset1) = ev.plus(data1(offset1), data2(offset2))
        }
      }
      Apply.apply2[T](self, t, func2)
      result
    }
  }

  def sub[@specialized(Float, Double) T: ClassTag](s: T, t: DenseTensor[T])
    (implicit ev: TensorNumeric[T]): Tensor[T] = {
    val result = new DenseTensor[T]()
    result.resizeAs(t)
    result.copy(t)
    val func = new TensorFunc2[T] {
      override def apply(data: Array[T], index: Int): Unit = {
        data(index) = ev.minus(data(index), s)
      }
    }
    Apply.apply1[T](result, func)
    result
  }

  def sub[@specialized(Float, Double) T: ClassTag](self: DenseTensor[T], t: Tensor[T])
    (implicit ev: TensorNumeric[T]): Tensor[T] = {
    val result = new DenseTensor[T]()
    result.resizeAs(self)
    result.copy(self)
    val func2 = new TensorFunc4[T] {
      override def apply(data1: Array[T], offset1: Int, data2: Array[T], offset2: Int): Unit = {
        data1(offset1) = ev.minus(data1(offset1), data2(offset2))
      }
    }
    Apply.apply2[T](result, t, func2)
    result
  }

  def neg[@specialized(Float, Double) T: ClassTag](self: DenseTensor[T])
    (implicit ev: TensorNumeric[T]): Tensor[T] = {
    val result = new DenseTensor[T]()
    result.resizeAs(self)
    result.copy(self)

    val func = new TensorFunc2[T] {
      override def apply(data: Array[T], index: Int): Unit = {
        data(index) = ev.negative(data(index))
      }
    }
    Apply.apply1[T](result, func)
    result
  }

  def divide[@specialized(Float, Double) T: ClassTag](s: T, t: DenseTensor[T])
    (implicit ev: TensorNumeric[T]): Tensor[T] = {
    val result = new DenseTensor[T]()
    result.resizeAs(t)
    result.copy(t)
    val func = new TensorFunc2[T] {
      override def apply(data: Array[T], index: Int): Unit = {
        data(index) = ev.divide(data(index), s)
      }
    }
    Apply.apply1[T](result, func)
    result
  }

  def divide[@specialized(Float, Double) T: ClassTag](self: DenseTensor[T], t: Tensor[T])
    (implicit ev: TensorNumeric[T]): Tensor[T] = {
    val result = new DenseTensor[T]()
    result.resizeAs(self)
    result.copy(self)
    val func2 = new TensorFunc4[T] {
      override def apply(data1: Array[T], offset1: Int, data2: Array[T], offset2: Int): Unit = {
        data1(offset1) = ev.divide(data1(offset1), data2(offset2))
      }
    }
    Apply.apply2[T](result, t, func2)
    result
  }

  def mul[@specialized(Float, Double) T: ClassTag](s: T, t: DenseTensor[T])
    (implicit ev: TensorNumeric[T]): Tensor[T] = {
    val result = new DenseTensor[T]()
    result.resizeAs(t)
    result.copy(t)
    val func = new TensorFunc2[T] {
      override def apply(data: Array[T], index: Int): Unit = {
        data(index) = ev.times(data(index), s)
      }
    }
    Apply.apply1[T](result, func)
    result
  }

  def mul[@specialized(Float, Double) T: ClassTag](self: Tensor[T], t: Tensor[T])
    (implicit ev: TensorNumeric[T]): Tensor[T] = {
    if (self.nDimension() == 1 && t.nDimension() == 1) {
      require(self.size(1) == t.size(1), "vector size not match")

      val result = ev.dot(self.nElement(), self.storage().array(), self.storageOffset() - 1,
        self.stride(1), t.storage().array(), t.storageOffset() - 1, t.stride(1))
      new DenseTensor(new ArrayStorage(Array(result)))
    } else if (self.nDimension() == 2 && t.nDimension() == 1) {
      val result = new DenseTensor[T](self.size(1))
      DenseTensorBLAS.gemv[T](ev.fromType[Int](1), self, t, ev.fromType[Int](0), result)
      result
    } else if (self.nDimension() == 2 && t.nDimension() == 2) {
      val result = new DenseTensor[T](t.size(2), self.size(1)).t()
      addmm[T](result, ev.fromType[Int](0), result, ev.fromType[Int](1), self, t)
      result
    } else {
      throw new UnsupportedOperationException(s"multiplication between ${self.nDimension()}D and " +
        s"${t.nDimension()}D not yet supported")
    }
  }

  def pow[@specialized(Float, Double) T: ClassTag](self: DenseTensor[T], x: Tensor[T], n: T)
    (implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(self.nElement() == x.nElement())
    if (MKL.isMKLLoaded && self.isContiguous() && x.isContiguous()) {
      ev.vPowx(self.nElement(), x.storage().array(), x.storageOffset() - 1, n,
        self.storage().array(), self.storageOffset() - 1)
    } else {
      val func = new TensorFunc4[T] {
        override def apply(data1: Array[T], offset1: Int, data2: Array[T], offset2: Int): Unit = {
          data1(offset1) = ev.pow(data2(offset2), n)
        }
      }
      DenseTensorApply.apply2[T](self, x, func)
    }
    self
  }

  def exp[@specialized(Float, Double) T: ClassTag](self: DenseTensor[T], x: Tensor[T])
    (implicit ev: TensorNumeric[T]): Tensor[T] = {
    if (self.nElement() != x.nElement()) {
      self.resizeAs(x)
    }

    if (MKL.isMKLLoaded && self.isContiguous() && x.isContiguous()) {
      ev.vExp(self.nElement(), x.storage().array(), x.storageOffset() - 1,
        self.storage().array(), self.storageOffset() - 1)
    } else {
      val func = new TensorFunc4[T] {
        override def apply(data1: Array[T], offset1: Int, data2: Array[T], offset2: Int): Unit = {
          data1(offset1) = ev.exp(data2(offset2))
        }
      }
      DenseTensorApply.apply2[T](self, x, func)
    }
    self
  }

  def log[@specialized(Float, Double) T: ClassTag](self: DenseTensor[T], x: Tensor[T])
    (implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(self.nElement() == x.nElement())
    if (MKL.isMKLLoaded && self.isContiguous() && x.isContiguous()) {
      ev.vLn(self.nElement(), x.storage().array(), x.storageOffset() - 1,
        self.storage().array(), self.storageOffset() - 1)
    } else {
      val func = new TensorFunc4[T] {
        override def apply(data1: Array[T], offset1: Int, data2: Array[T], offset2: Int): Unit = {
          data1(offset1) = ev.log(data2(offset2))
        }
      }
      DenseTensorApply.apply2[T](self, x, func)
    }
    self
  }

  def sqrt[@specialized(Float, Double) T: ClassTag](self: DenseTensor[T], x: Tensor[T])
    (implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(self.nElement() == x.nElement())
    if (MKL.isMKLLoaded && self.isContiguous() && x.isContiguous()) {
      ev.vSqrt(self.nElement(), x.storage().array(), x.storageOffset() - 1,
        self.storage().array(), self.storageOffset() - 1)
    } else {
      val func = new TensorFunc4[T] {
        override def apply(data1: Array[T], offset1: Int, data2: Array[T], offset2: Int): Unit = {
          data1(offset1) = ev.sqrt(data2(offset2))
        }
      }
      DenseTensorApply.apply2[T](self, x, func)
    }
    self
  }

  def tanh[@specialized(Float, Double) T: ClassTag](self: DenseTensor[T], x: Tensor[T])
                                                   (implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(self.nElement() == x.nElement())
    if (MKL.isMKLLoaded && self.isContiguous() && x.isContiguous()) {
      ev.vTanh(self.nElement(), x.storage().array(), x.storageOffset() - 1,
        self.storage().array(), self.storageOffset() - 1)
    } else {
      val func = new TensorFunc4[T] {
        override def apply(data1: Array[T], offset1: Int, data2: Array[T], offset2: Int): Unit = {
          data1(offset1) = ev.tanh(data2(offset2))
        }
      }
      DenseTensorApply.apply2[T](self, x, func)
    }
    self
  }

  def log1p[@specialized(Float, Double) T: ClassTag](self: DenseTensor[T], x: Tensor[T])
    (implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(self.nElement() == x.nElement())
    if (MKL.isMKLLoaded && self.isContiguous() && x.isContiguous()) {
      ev.vLog1p(self.nElement(), x.storage().array(), x.storageOffset() - 1,
        self.storage().array(), self.storageOffset() - 1)

    } else {
      val func = new TensorFunc4[T] {
        override def apply(data1: Array[T], offset1: Int, data2: Array[T], offset2: Int): Unit = {
          data1(offset1) = ev.log1p(data2(offset2))
        }
      }
      DenseTensorApply.apply2[T](self, x, func)

    }
    self
  }

  def prodAll[@specialized(Float, Double) T](self: DenseTensor[T])(
    implicit ev: TensorNumeric[T]): T = {
    var product = ev.fromType[Int](1)
    val func = new TensorFunc2[T] {
      override def apply(data: Array[T], index: Int): Unit = {
        product = ev.times(data(index), product)
      }
    }
    Apply.apply1[T](self, func)
    product
  }

  def sumAll[@specialized(Float, Double) T](self: DenseTensor[T])(
    implicit ev: TensorNumeric[T]): T = {
    var sum = ev.fromType[Int](0)
    val func = new TensorFunc2[T] {
      override def apply(data: Array[T], index: Int): Unit = {
        sum = ev.plus(data(index), sum)
      }
    }
    Apply.apply1[T](self, func)
    sum
  }

  def prod[@specialized(Float, Double) T: ClassTag](self: DenseTensor[T], x: Tensor[T], _dim: Int)
    (implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(_dim >= 0 && _dim < x.nDimension, s"dimension ${_dim + 1} out of range")
    val result = if (self == null) new DenseTensor[T]() else self
    val sizes = x.size()
    sizes(_dim) = 1
    result.resize(sizes)
    DenseTensorDimApply.dimApply2[T](result, x, _dim,
      (rData, rOffset, rStride, rSize, tData, tOffset, tStride, tSize) => {
        rData(rOffset) = ev.prod(tSize, tData, tOffset, tStride)
      })

    result
  }

  def sum[@specialized T: ClassTag](self: DenseTensor[T], x: Tensor[T], _dim: Int)
    (implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(_dim >= 0 && _dim < x.nDimension, s"dimension ${_dim + 1} out of range")
    val result = if (self == null) new DenseTensor[T]() else self
    val sizes = x.size()
    sizes(_dim) = 1
    result.resize(sizes)
    DenseTensorDimApply.dimApply2[T](result, x, _dim,
      (rData, rOffset, rStride, rSize, tData, tOffset, tStride, tSize) => {
        rData(rOffset) = ev.sum(tSize, tData, tOffset, tStride)
      })

    result
  }

  def maxAll[@specialized(Float, Double) T](self: DenseTensor[T])(
    implicit ev: TensorNumeric[T]): T = {
    var max = ev.fromType[Int](0)
    var first = true
    val func = new TensorFunc2[T] {
      override def apply(data: Array[T], index: Int): Unit = {
        if (first) {
          first = false
          max = data(index)
        } else if (ev.isGreater(data(index), max)) {
          max = data(index)
        }
      }
    }
    Apply.apply1[T](self, func)
    max
  }

  def minAll[@specialized(Float, Double) T](self: DenseTensor[T])(
    implicit ev: TensorNumeric[T]): T = {
    var min = ev.fromType[Int](Int.MaxValue)
    var first = true
    val func = new TensorFunc2[T] {
      override def apply(data: Array[T], index: Int): Unit = {
        if (first) {
          first = false
          min = data(index)
        } else if (ev.isGreater(min, data(index))) {
          min = data(index)
        }
      }
    }
    Apply.apply1[T](self, func)
    min
  }

  def addmm[@specialized(Float, Double) T: ClassTag](r: Tensor[T], beta: T, t: Tensor[T],
    alpha: T, m1: Tensor[T], m2: Tensor[T])
    (implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(m1.dim() == 2 && m2.dim() == 2,
      s"matrices expected, got ${m1.dim()}, ${m2.dim()} tensors")
    require(m1.size(2) == m2.size(1),
      s"size mismatch, m1:${m1.size().mkString("x")} m2:${m2.size().mkString("x")}")
    require(t.dim() == 2,
      s"matrix expected, got ${t.dim()} tensor for t")
    require(t.size(1) == m1.size(1) && t.size(2) == m2.size(2),
      s"size mismatch. t:${t.size().mkString("x")}, " +
        s"m1:${m1.size().mkString("x")} + m2:${m2.size().mkString("x")}")

    if (r != t) {
      r.resizeAs(t)
      r.copy(t)
    }

    var _r: Tensor[T] = null
    var _m1: Tensor[T] = m1
    var _m2: Tensor[T] = m2
    var transpose_r = ' '
    if (r.stride(1) == 1 && r.stride(2) != 0) {
      transpose_r = 'n'
      _r = r
    } else if (r.stride(2) == 1 && r.stride(1) != 0) {
      val swap = _m2
      _m2 = _m1
      _m1 = swap
      transpose_r = 't'
      _r = r
    } else {
      transpose_r = 'n'
      _r = new DenseTensor[T](r.size(2), r.size(1))
      _r.copy(r)
      _r = _r.transpose(1, 2)
    }

    val index1 = if (transpose_r == 'n') 1 else 2
    val index2 = if (transpose_r == 'n') 2 else 1
    var transpose_m1 = ' '
    var __m1: Tensor[T] = null
    if (_m1.stride(index1) == 1 && _m1.stride(index2) != 0) {
      transpose_m1 = 'n'
      __m1 = _m1
    } else if (_m1.stride(index2) == 1 && _m1.stride(index1) != 0) {
      transpose_m1 = 't'
      __m1 = _m1
    } else {
      transpose_m1 = if (transpose_r == 'n') 't' else 'n'
      __m1 = _m1.contiguous()
    }

    var transpose_m2 = ' '
    var __m2: Tensor[T] = null
    if (_m2.stride(index1) == 1 && _m2.stride(index2) != 0) {
      transpose_m2 = 'n'
      __m2 = _m2
    } else if (_m2.stride(index2) == 1 && _m2.stride(index1) != 0) {
      transpose_m2 = 't'
      __m2 = _m2
    } else {
      transpose_m2 = if (transpose_r == 'n') 't' else 'n'
      __m2 = _m2.contiguous()
    }

    DenseTensorBLAS.gemm[T](transpose_m1, transpose_m2, _r.size(index1), _r.size(index2),
      __m1.size(index2), alpha, __m1.storage().array(), __m1.storageOffset() - 1,
      if (transpose_m1 == 'n') __m1.stride(index2) else __m1.stride(index1),
      __m2.storage().array(), __m2.storageOffset() - 1,
      if (transpose_m2 == 'n') __m2.stride(index2) else __m2.stride(index1),
      beta,
      _r.storage().array(), _r.storageOffset() - 1,
      _r.stride(index2)
    )
    if (_r != r) {
      r.copy(_r)
    }
    r
  }

  def addr[@specialized(Float, Double) T](r: Tensor[T], beta: T, t: Tensor[T],
    alpha: T, vec1: Tensor[T], vec2: Tensor[T])(implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(vec1.dim() == 1 && vec2.dim() == 1)
    require(t.dim() == 2)
    require(t.size(1) == vec1.size(1) && t.size(2) == vec2.size(1))

    if (!r.eq(t)) {
      r.resizeAs(t).copy(t)
    }

    if (beta != 1) {
      r.mul(beta)
    }

    if (r.stride(1) == 1) {
      val lda = if (t.stride(2) == 1) {
        r.size(1)
      } else {
        r.stride(2)
      }
      ev.ger(vec1.size(1), vec2.size(1), alpha, vec1.storage().array(), vec1.storageOffset() - 1,
        vec1.stride(1), vec2.storage().array(), vec2.storageOffset() - 1, vec2.stride(1),
        r.storage().array(), r.storageOffset() - 1, lda)
    } else if (r.stride(2) == 1) {
      ev.ger(vec2.size(1), vec1.size(1), alpha, vec2.storage().array(), vec2.storageOffset() - 1,
        vec2.stride(1), vec1.storage().array(), vec1.storageOffset() - 1, vec1.stride(1),
        r.storage().array(), r.storageOffset() - 1, r.stride(1))
    } else {
      val cr = r.contiguous()
      ev.ger(vec2.size(1), vec1.size(1), alpha, vec2.storage().array(), vec2.storageOffset() - 1,
        vec2.stride(1), vec1.storage().array(), vec1.storageOffset() - 1, vec1.stride(1),
        cr.storage().array(), cr.storageOffset() - 1, cr.stride(1))
      r.copy(cr)
    }

    r
  }

  def baddbmm[@specialized(Float, Double) T: ClassTag]
  (result: Tensor[T], beta: T, M: Tensor[T], alpha: T, batch1: Tensor[T], batch2: Tensor[T])
    (implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(batch1.dim() == 3, s"expected 3D tensor, got ${batch1.dim()}D")
    require(batch2.dim() == 3, s"expected 3D tensor, got ${batch2.dim()}D")
    require(batch1.size(1) == batch2.size(1), "equal number of batches expected, got " +
      s"${batch1.size(1)}, ${batch2.size(1)}")
    require(batch1.size(3) == batch2.size(2), s"wrong matrix size, batch1: " +
      s"${batch1.size(2)}${batch1.size(3)}, batch2: " +
      s"${batch2.size(2)}${batch2.size(3)}")

    val bs = batch1.size(1)
    val dim1 = batch1.size(2)
    val dim2 = batch2.size(3)
    require(M.size(1) == bs, "output tensor of incorrect size")
    require(M.size(2) == dim1, "output tensor of incorrect size")
    require(M.size(3) == dim2, "output tensor of incorrect size")

    if (M != result) {
      result
        .resizeAs(M)
        .copy(M)
    }

    var batch = 1
    while (batch <= batch1.size(1)) {
      val m1 = batch1.select(1, batch)
      val m2 = batch2.select(1, batch)
      val resultMatrix = result.select(1, batch)

      addmm(resultMatrix, beta, resultMatrix, alpha, m1, m2)
      batch += 1
    }

    result
  }

  def addmv[@specialized(Float, Double) T](r: Tensor[T], beta: T, t: Tensor[T], alpha: T,
    mat: Tensor[T], vec: Tensor[T])(implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(mat.nDimension() == 2 && vec.nDimension() == 1)
    require(mat.size(2) == vec.size(1))
    require(t.nDimension() == 1)
    require(t.size(1) == mat.size(1), s"${t.size(1)} == ${mat.size(1)}")

    if (!r.eq(t)) {
      r.resizeAs(t).copy(t)
    }

    if (mat.stride(1) == 1) {
      val lda = if (mat.size(2) == 1) {
        mat.size(1)
      } else {
        mat.stride(2)
      }
      ev.gemv('N', mat.size(1), mat.size(2), alpha, mat.storage().array(), mat.storageOffset() - 1,
        lda, vec.storage().array(), vec.storageOffset() - 1, vec.stride(1), beta,
        r.storage().array(),
        r.storageOffset() - 1, r.stride(1))
    } else if (mat.stride(2) == 1) {
      ev.gemv('T', mat.size(2), mat.size(1), alpha, mat.storage().array(), mat.storageOffset() - 1,
        mat.stride(1), vec.storage().array(), vec.storageOffset() - 1, vec.stride(1), beta,
        r.storage().array(), r.storageOffset() - 1, r.stride(1))
    } else {
      val cmat = mat.contiguous()
      ev.gemv('T', cmat.size(2), cmat.size(1), alpha, cmat.storage().array(),
        cmat.storageOffset() - 1, cmat.stride(1), vec.storage().array(), vec.storageOffset() - 1,
        vec.stride(1), beta, r.storage().array(), r.storageOffset() - 1, r.stride(1))
    }

    r
  }

  def meanAll[@specialized(Float, Double) T](self: DenseTensor[T])(
    implicit ev: TensorNumeric[T]): T = {
    var sum = ev.fromType[Int](0)
    val func = new TensorFunc2[T] {
      override def apply(data: Array[T], index: Int): Unit = {
        sum = ev.plus(data(index), sum)
      }
    }
    Apply.apply1[T](self, func)
    ev.divide(sum, ev.fromType[Int](self.nElement()))
  }

  def mean[@specialized(Float, Double) T: ClassTag](self: DenseTensor[T], _dim: Int)(
    implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(_dim >= 0 && _dim < self.nDimension, s"dimension ${_dim + 1} out of range")
    val result = new DenseTensor[T]()
    val sizes = self.size()
    sizes(_dim) = 1
    DenseTensor.resize(result, sizes)
    DenseTensorDimApply.dimApply2[T](result, self, _dim,
      (rData, rOffset, rStride, rSize, tData, tOffset, tStride, tSize) => {
        var sum = ev.fromType[Int](0)
        var i = 0
        while (i < tSize) {
          sum = ev.plus(sum, tData(tOffset + i * tStride))
          i += 1
        }
        rData(rOffset) = ev.divide(sum, ev.fromType[Int](self.size(_dim + 1)))
      })
    result
  }

  /**
   * returns the p-norms of the Tensor x computed over the dimension dim.
 *
   * @param self
   * @param value value-norms
   * @param _dim the dimension dim
   * @return
   */
  def norm[@specialized(Float, Double) T: ClassTag](self: DenseTensor[T], result: Tensor[T],
   value: Int, _dim: Int)(implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(_dim >= 0 && _dim < self.nDimension, "invalid dimension")
    val sizes = self.size()
    sizes(_dim) = 1
    result.resize(sizes)

    if (value == 0) {
      DenseTensorDimApply.dimApply2[T](self, result, _dim,
        (rData, rOffset, rStride, rSize, tData, tOffset, tStride, tSize) => {
          var sum = ev.fromType[Int](0)
          var i = 0
          while (i < rSize) {
            sum = ev.plus(sum, rData(rOffset + i * rStride))
            i += 1
          }
          tData(tOffset) = sum
        })
    } else {
      DenseTensorDimApply.dimApply2[T](self, result, _dim,
        (rData, rOffset, rStride, rSize, tData, tOffset, tStride, tSize) => {
          var sum = ev.fromType[Int](0)
          var i = 0
          while (i < rSize) {
            sum = ev.plus(sum, ev.pow(ev.abs(rData(rOffset + i * rStride)), ev.fromType(value)))
            i += 1
          }
          tData(tOffset) = ev.pow(sum, ev.fromType(1.0 / value))
        })
    }
    result
  }

  def nearlyEqual[@specialized(Float, Double) T](a: T, b: T, epsilon: Double)(
    implicit ev: TensorNumeric[T]): Boolean = {
    ev.nearlyEqual(a, b, epsilon)
  }

  def cmax[@specialized(Float, Double) T](self: DenseTensor[T], x: Tensor[T], y: Tensor[T])
                                         (implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(self.nElement() == y.nElement() && self.nElement() == x.nElement(),
      "element number doesn't match")
    // todo: the performance of contiguous tensor should be optimized
    val func = new TensorFunc6[T] {
      override def apply(data1: Array[T], offset1: Int, data2: Array[T], offset2: Int,
                         data3: Array[T], offset3: Int): Unit = {
        data1(offset1) = ev.max(data2(offset2), data3(offset3))
      }
    }
    Apply.apply3[T](self, x, y, func)
    self
  }
  def cmin[@specialized(Float, Double) T](self: DenseTensor[T], x: Tensor[T], y: Tensor[T])
                                         (implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(self.nElement() == y.nElement() && self.nElement() == x.nElement(),
      "element number doesn't match")
    // todo: the performance of contiguous tensor should be optimized
    val func = new TensorFunc6[T] {
      override def apply(data1: Array[T], offset1: Int, data2: Array[T], offset2: Int,
                         data3: Array[T], offset3: Int): Unit = {
        data1(offset1) = ev.min(data2(offset2), data3(offset3))
      }
    }
    Apply.apply3[T](self, x, y, func)
    self
  }


  val doubleEpsilon = System.getProperty("DoubleTensorEpsilon", "0.0000001").toDouble
  val floatEpsilon = System.getProperty("FloatTensorEpsilon", "0.00001").toDouble
}

