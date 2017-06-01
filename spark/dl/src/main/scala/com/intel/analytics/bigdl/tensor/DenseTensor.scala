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

import breeze.linalg.{DenseMatrix => BrzDenseMatrix, DenseVector => BrzDenseVector}
import com.intel.analytics.bigdl.mkl.MKL
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.{File, Table}
import org.apache.spark.mllib.linalg.{DenseMatrix, DenseVector, Matrix, Vector}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

@SerialVersionUID(5876322619614900645L)
private[tensor] class DenseTensor[@specialized(Float, Double) T: ClassTag](
  private[tensor] var _storage: Storage[T],
  private[tensor] var _storageOffset: Int,
  private[tensor] var _size: Array[Int],
  private[tensor] var _stride: Array[Int],
  var nDimension: Int)(implicit ev: TensorNumeric[T])
  extends Tensor[T] {

  override def storage(): Storage[T] = _storage

  override def storageOffset(): Int = _storageOffset + 1

  override def dim(): Int = nDimension

  override def nElement(): Int = {
    if (this.nDimension == 0) {
      0
    } else {
      var n = 1
      var d = 0
      while (d < this.nDimension) {
        n = n * this._size(d)
        d += 1
      }
      n
    }
  }

  override def squeeze(): Tensor[T] = DenseTensor.squeeze(this)

  override def squeeze(dim: Int): Tensor[T] = DenseTensor.squeeze(this, dim - 1)

  override def squeezeNewTensor(): Tensor[T] = {
    val result = new DenseTensor(this.storage(), this.storageOffset(), this._size, this._stride)
    result.squeeze()
  }

  override def size(): Array[Int] = _size.slice(0, this.nDimension)

  override def size(dim: Int): Int = {
    require(dim > 0 && dim <= this.nDimension,
      s"dimension ${dim} out of range of ${this.nDimension}D tensor")
    _size(dim - 1)
  }

  override def stride(): Array[Int] = _stride.slice(0, this.nDimension)

  override def stride(dim: Int): Int = {
    require(dim > 0 && dim <= this.nDimension,
      s"dimension ${dim} out of range of ${this.nDimension}D tensor")
    _stride(dim - 1)
  }

  override def resizeAs(src: Tensor[_]): Tensor[T] = {
    DenseTensor.resizeAs(this, src)
    this
  }

  override def resize(sizes: Array[Int], strides: Array[Int]): Tensor[T] = {
    DenseTensor.resize(this, sizes, strides)
    this
  }

  override def resize(size1: Int): Tensor[T] = {
    if (this.nDimension != 1 || this.size(1) != size1) {
      DenseTensor.resize(this, Array(size1))
    } else {
      this
    }
  }

  override def resize(size1: Int, size2: Int): Tensor[T] = {
    if (this.nDimension != 2 || this.size(1) != size1 || this.size(2) != size2) {
      DenseTensor.resize(this, Array(size1, size2))
    } else {
      this
    }
  }

  override def resize(size1: Int, size2: Int, size3: Int): Tensor[T] = {
    if (this.nDimension != 3 || this.size(1) != size1 || this.size(2) != size2 ||
      this.size(3) != size3) {
      DenseTensor.resize(this, Array(size1, size2, size3))
    } else {
      this
    }
  }

  override def resize(size1: Int, size2: Int, size3: Int, size4: Int): Tensor[T] = {
    if (this.nDimension != 4 || this.size(1) != size1 || this.size(2) != size2 ||
      this.size(3) != size3 ||
      this.size(4) != size4) {
      DenseTensor.resize(this, Array(size1, size2, size3, size4))
    } else {
      this
    }
  }

  override def resize(size1: Int, size2: Int, size3: Int, size4: Int, size5: Int): Tensor[T] = {
    if (this.nDimension != 5 || this.size(1) != size1 || this.size(2) != size2 ||
      this.size(3) != size3 || this.size(4) != size4 || this.size(5) != size5) {
      DenseTensor.resize(this, Array(size1, size2, size3, size4, size5))
    } else {
      this
    }
  }

  override def view(sizes: Array[Int]): Tensor[T] = {
    require(this.isContiguous(), "current tensor is not contiguous")
    require(sizes.product == this.nElement(), "invalid size eElement")

    new DenseTensor(this.storage(), this.storageOffset(), sizes.clone())
  }

  override def unfold(dim: Int, size: Int, step: Int): Tensor[T] = {
    require(this.nDimension > 0, "cannot unfold an empty tensor")
    require(dim > 0 && dim <= this.nDimension, "out of range")
    require(size <= this.size(dim), "out of range")
    require(step > 0, "invalid step")

    val newTensor = this

    val newSize = new Array[Int](this.nDimension + 1)
    val newStride = new Array[Int](this.nDimension + 1)

    newSize(this.nDimension) = size
    newStride(this.nDimension) = this.stride(dim)

    var d = 0
    while (d < this.nDimension) {
      if (d + 1 == dim) {
        newSize(d) = (this.size(d + 1) - size) / step + 1
        newStride(d) = step * this.stride(d + 1)
      } else {
        newSize(d) = this.size(d + 1)
        newStride(d) = this.stride(d + 1)
      }
      d = d + 1
    }

    new DenseTensor(this._storage, this._storageOffset, newSize, newStride, this.dim() + 1)
  }

  private[tensor] def this(d1: Int)(implicit ev: TensorNumeric[T]) =
    this(new ArrayStorage[T](new Array[T](d1)), 0, Array(d1),
      Array(1), 1)

  private[tensor] def this(d1: Int, d2: Int)(implicit ev: TensorNumeric[T]) =
    this(new ArrayStorage[T](new Array[T](d1 * d2)), 0, Array(d1, d2),
      Array(d2, 1), 2)

  private[tensor] def this(d1: Int, d2: Int, d3: Int)(implicit ev: TensorNumeric[T]) =
    this(new ArrayStorage[T](new Array[T](d1 * d2 * d3)), 0, Array(d1, d2, d3),
      Array(d3 * d2, d3, 1), 3)

  private[tensor] def this(d1: Int, d2: Int, d3: Int, d4: Int)(implicit ev: TensorNumeric[T]) =
    this(new ArrayStorage[T](new Array[T](d1 * d2 * d3 * d4)), 0, Array(d1, d2, d3, d4),
      Array(d4 * d3 * d2, d4 * d3, d4, 1), 4)

  private[tensor] def this(d1: Int, d2: Int, d3: Int, d4: Int, d5: Int)(
    implicit ev: TensorNumeric[T]) =
    this(new ArrayStorage[T](new Array[T](d1 * d2 * d3 * d4 * d5)), 0, Array(d1, d2, d3, d4, d5),
      Array(d5 * d4 * d3 * d2, d5 * d4 * d3, d5 * d4, d5, 1), 5)

  private[tensor] def this(dims: Int*)(implicit ev: TensorNumeric[T]) =
    this(new ArrayStorage[T](new Array[T](dims.product)), 0, dims.toArray,
      DenseTensor.size2Stride(dims.toArray), dims.length)

  private[tensor] def this(storage: Storage[T])(implicit ev: TensorNumeric[T]) = {
    this(null, 0, null, null, 0)
    val _storageOffset = 0
    val _size = Array(storage.length)
    val _stride = Array(1)
    DenseTensor.newWithStorage(this, storage, _storageOffset, _size, _stride, ev)
  }

  private[tensor] def this(storage: Storage[T], storageOffset: Int, size: Array[Int] = null,
    stride: Array[Int] = null)(implicit ev: TensorNumeric[T]) = {
    this(null, 0, null, null, 0)
    if (storage != null) {
      val _storageOffset = storageOffset - 1
      val _size = if (size == null) Array(storage.length) else size
      val _stride = if (size == null) null else stride
      DenseTensor.newWithStorage(this, storage, _storageOffset, _size, _stride, ev)
    }
  }

  private[tensor] def this(other: Tensor[T])(implicit ev: TensorNumeric[T]) = {
    this(null, 0, null, null, 0)
    val _storage = other.storage()
    val _storageOffset = other.storageOffset() - 1
    val _size = other.size()
    val _stride = other.stride()
    DenseTensor.newWithStorage(this, _storage, _storageOffset, _size, _stride, ev)
  }

  private[tensor] def this()(implicit ev: TensorNumeric[T]) = this(null, 0, null, null, 0)

  override def fill(v: T): Tensor[T] = {
    if (this.isContiguous()) {
      this.storage().fill(v, this.storageOffset(), this.nElement())
    } else {
      val func = new TensorFunc2[T] {
        override def apply(data: Array[T], index: Int): Unit = {
          data(index) = v
        }
      }
      DenseTensorApply.apply1[T](this, func)
    }
    this
  }

  override def zero(): Tensor[T] = {
    this.fill(ev.fromType[Int](0))
  }

  override def randn(): Tensor[T] = {
    if (this.isContiguous()) {
      var i = 0
      val total = this.nElement()
      val data = this.storage().array()
      val offset = this.storageOffset() - 1
      while (i < total) {
        data(offset + i) = ev.randn()
        i += 1
      }
    } else {
      val func = new TensorFunc2[T] {
        override def apply(data: Array[T], index: Int): Unit = {
          data(index) = ev.randn()
        }
      }
      DenseTensorApply.apply1[T](this, func)
    }
    this
  }

  override def bernoulli(p: Double): Tensor[T] = {

    if (this.isContiguous()) {
      var i = 0
      val total = this.nElement()
      val data = this.storage().array()
      val offset = this.storageOffset() - 1
      while (i < total) {
        data(offset + i) = if (RNG.bernoulli(p)) {
          ev.fromType[Int](1)
        } else {
          ev.fromType[Int](0)
        }
        i += 1
      }
    } else {
      val func = new TensorFunc2[T] {
        override def apply(data: Array[T], index: Int): Unit = {
          data(index) =
            if (RNG.bernoulli(p)) {
              ev.fromType[Int](1)
            } else {
              ev.fromType[Int](0)
            }
        }
      }
      DenseTensorApply.apply1[T](this, func)
    }
    this
  }


  override def rand(): Tensor[T] = {
    if (this.isContiguous()) {
      var i = 0
      val total = this.nElement()
      val data = this.storage().array()
      val offset = this.storageOffset() - 1
      while (i < total) {
        data(offset + i) = ev.rand()
        i += 1
      }
    } else {
      val func = new TensorFunc2[T] {
        override def apply(data: Array[T], index: Int): Unit = {
          data(index) = ev.rand()
        }
      }
      DenseTensorApply.apply1[T](this, func)
    }
    this
  }

  override def set(other: Tensor[T]): Tensor[T] = {
    DenseTensor.rawSet(this, other.storage(), other.storageOffset() - 1, other.nDimension(),
      other.size(), other.stride())
  }

  override def set(storage: Storage[T], storageOffset: Int = 1, sizes: Array[Int] = null,
    strides: Array[Int] = null): Tensor[T] = {
    if (sizes != null && strides != null) {
      require(sizes.length == strides.length)
    }

    DenseTensor.rawSet(this, storage, storageOffset - 1,
      if (sizes == null) 0 else sizes.length,
      sizes, strides)
  }

  override def set(): Tensor[T] = {
    this.resize(0)
    if(this._storage != null) {
      this._storage.resize(0)
    }
    this
  }

  override def transpose(dim1: Int, dim2: Int): Tensor[T] = {
    val result = DenseTensor.newWithTensor(this)
    DenseTensor.transpose(result, null, dim1 - 1, dim2 - 1)
    result
  }

  override def t(): Tensor[T] = {
    require(this.nDimension == 2, "t() is only for 2D tensor")
    transpose(1, 2)
  }

  override def select(dim: Int, index: Int): Tensor[T] = {
    val _dimension = dim - 1
    val _sliceIndex = index - 1

    if (this.nDimension > 1) {
      val result = DenseTensor.newWithTensor(this)
      DenseTensor.select(result, null, _dimension, _sliceIndex)
      result
    } else {
      require(this.nDimension == 1, "empty tensor")
      DenseTensor.get1dTensor(this, _sliceIndex)
    }
  }

  override def clone(): Tensor[T] = {
    DenseTensor.newClone(this)
  }

  override def copy(other: Tensor[T]): Tensor[T] = {
    DenseTensor.copy(this, other)
    this
  }

  override def narrow(dim: Int, index: Int, size: Int): Tensor[T] = {
    val result = DenseTensor.newWithTensor(this)
    DenseTensor.narrow(result, null, dim - 1, index - 1, size)
    result
  }

  override def apply1(func: T => T): Tensor[T] = {
    val func2 = new TensorFunc2[T] {
      override def apply(data: Array[T], index: Int): Unit = {
        data(index) = func(data(index))
      }
    }
    DenseTensorApply.apply1[T](this, func2)
    this
  }

  override def map(other: Tensor[T], func: (T, T) => T): Tensor[T] = {
    val func2 = new TensorFunc4[T] {
      override def apply(data1: Array[T], index1: Int, data2: Array[T], index2: Int): Unit = {
        data1(index1) = func(data1(index1), data2(index2))
      }
    }
    DenseTensorApply.apply2[T](this, other, func2)
    this
  }

  override def apply(index: Int): Tensor[T] = {
    require(this.nDimension > 0, "empty tensor")
    var _index = index - 1
    if (_index < 0) _index = this._size(0) + _index + 1
    require(_index >= 0 && _index < this._size(0),
      s"out of range, ${_index}: 0 to ${this._size(0)}")

    if (this.nDimension == 1) {
      this.narrow(1, index, 1)
    } else {
      val result = DenseTensor.newWithTensor(this)
      DenseTensor.select(result, null, 0, _index)
      result
    }
  }

  override def apply(table: Table): Tensor[T] = {
    val (tensor, offset) = subset(table)
    offset match {
      case Some(i) =>
        val result = new DenseTensor[T](1)
        result.setValue(1, tensor.storage()(i))
        result
      case None => tensor
    }
  }

  override def update(table: Table, value: T): Unit = {
    val (tensor, offset) = subset(table)
    offset match {
      case Some(i) => tensor.storage()(i) = value
      case None => tensor.fill(value)
    }
  }

  override def update(table: Table, src: Tensor[T]): Unit = {
    val (tensor, offset) = subset(table)
    tensor.copy(src)
  }

  override def update(index: Int, src: Tensor[T]): Unit = {
    require(this.nDimension > 0, "empty tensor")
    var _index = index - 1
    if (_index < 0) _index = this._size(0) + _index + 1
    require(_index >= 0 && _index < this._size(0), "out of range")
    val tensor = DenseTensor.newWithTensor(this)
    DenseTensor.narrow(tensor, null, 0, _index, 1)
    tensor.copy(src)
  }

  private def subset(table: Table): (Tensor[T], Option[Int]) = {
    var cdim = 0
    require(table.length <= this.nDimension, "too many indices provided")
    val tensor = DenseTensor.newWithTensor(this)
    var d = 1
    while (d <= table.length) {
      table[Any](d) match {
        case index: Int =>
          var z = index - 1
          if (z < 0) z = tensor._size(cdim) + z + 1
          require(z >= 0 && z < tensor._size(cdim), "index out of bound")
          if (tensor.nDimension == 1) {
            return (tensor, Some(tensor._storageOffset + z * tensor._stride(0)))
          } else {
            DenseTensor.select(tensor, null, cdim, z)
          }
        case range: Table =>
          var start = 0
          var end = tensor._size(cdim) - 1
          if (range.length >= 1) {
            range[Any](1) match {
              case left: Int =>
                start = left - 1
            }
            end = start
          }

          if (start < 0) start = tensor._size(cdim) + start + 1
          require(start >= 0 && start < tensor._size(cdim), "start index out of bound")
          if (range.length >= 2) {
            range[Any](2) match {
              case right: Int =>
                end = right - 1
            }
          }
          if (end < 0) end = tensor._size(cdim) + end + 1
          require(end >= 0 && end < tensor._size(cdim), "end index out of bound")

          require(end >= start, "end index must be greater or equal to start index")
          DenseTensor.narrow(tensor, null, cdim, start, end - start + 1)
          cdim = cdim + 1
      }
      d += 1
    }

    (tensor, None)
  }

  override def apply(indexes: Array[Int]): T = {
    require(indexes.length == this.nDimension, "invalid size")
    var offset = this._storageOffset
    var d = 0
    while (d < indexes.length) {
      offset += getOffset(indexes(d) - 1, d + 1)
      d += 1
    }
    this._storage(offset)
  }

  override def valueAt(d1: Int): T = {
    require(1 == this.nDimension, s"invalid size: 1 == ${this.nDimension}")
    var offset = this._storageOffset
    offset += getOffset(d1 - 1, 1)
    this._storage(offset)
  }

  override def valueAt(d1: Int, d2: Int): T = {
    require(2 == this.nDimension, "invalid size")
    var offset = this._storageOffset
    offset += getOffset(d1 - 1, 1)
    offset += getOffset(d2 - 1, 2)
    this._storage(offset)
  }

  override def valueAt(d1: Int, d2: Int, d3: Int): T = {
    require(3 == this.nDimension, "invalid size")
    var offset = this._storageOffset
    offset += getOffset(d1 - 1, 1)
    offset += getOffset(d2 - 1, 2)
    offset += getOffset(d3 - 1, 3)
    this._storage(offset)
  }

  override def valueAt(d1: Int, d2: Int, d3: Int, d4: Int): T = {
    require(4 == this.nDimension, "invalid size")
    var offset = this._storageOffset
    offset += getOffset(d1 - 1, 1)
    offset += getOffset(d2 - 1, 2)
    offset += getOffset(d3 - 1, 3)
    offset += getOffset(d4 - 1, 4)
    this._storage(offset)
  }

  override def valueAt(d1: Int, d2: Int, d3: Int, d4: Int, d5: Int): T = {
    require(5 == this.nDimension, "invalid size")
    var offset = this._storageOffset
    offset += getOffset(d1 - 1, 1)
    offset += getOffset(d2 - 1, 2)
    offset += getOffset(d3 - 1, 3)
    offset += getOffset(d4 - 1, 4)
    offset += getOffset(d5 - 1, 5)
    this._storage(offset)
  }

  private def getOffset(z: Int, dim: Int): Int = {
    var _z = z
    if (_z < 0) {
      _z = this.size(dim) + _z + 1
    }
    require(_z >= 0 && _z < this.size(dim), "index out of bound")
    _z * this.stride(dim)
  }

  override def update(index: Int, value: T): Unit = {
    require(this.nDimension > 0, "empty tensor")
    var _index = index - 1
    if (_index < 0) _index = this._size(0) + _index + 1
    require(_index >= 0 && _index < this._size(0), "out of range")
    if (this.nDimension == 1) {
      _storage(this._storageOffset + _index * this._stride(0)) = value
    } else {
      val tensor = DenseTensor.newWithTensor(this)
      DenseTensor.narrow(tensor, null, 0, _index, 1)
      tensor.fill(value)
    }
  }

  override def update(indexes: Array[Int], value: T): Unit = {
    require(indexes.length == this.nDimension, "invalid size")
    var offset = this._storageOffset
    var d = 0
    while (d < indexes.length) {
      offset += getOffset(indexes(d) - 1, d + 1)
      d += 1
    }
    this._storage(offset) = value
  }

  override def setValue(d1: Int, d2: Int, d3: Int, d4: Int, value: T): this.type = {
    require(4 == this.nDimension, "invalid size")
    var offset = this._storageOffset
    offset += getOffset(d1 - 1, 1)
    offset += getOffset(d2 - 1, 2)
    offset += getOffset(d3 - 1, 3)
    offset += getOffset(d4 - 1, 4)
    this._storage(offset) = value
    this
  }

  override def setValue(d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, value: T): this.type = {
    require(5 == this.nDimension, "invalid size")
    var offset = this._storageOffset
    offset += getOffset(d1 - 1, 1)
    offset += getOffset(d2 - 1, 2)
    offset += getOffset(d3 - 1, 3)
    offset += getOffset(d4 - 1, 4)
    offset += getOffset(d5 - 1, 5)
    this._storage(offset) = value
    this
  }

  override def setValue(d1: Int, d2: Int, d3: Int, value: T): this.type = {
    require(3 == this.nDimension, "invalid size")
    var offset = this._storageOffset
    offset += getOffset(d1 - 1, 1)
    offset += getOffset(d2 - 1, 2)
    offset += getOffset(d3 - 1, 3)
    this._storage(offset) = value
    this
  }

  override def setValue(d1: Int, d2: Int, value: T): this.type = {
    require(2 == this.nDimension, "invalid size")
    var offset = this._storageOffset
    offset += getOffset(d1 - 1, 1)
    offset += getOffset(d2 - 1, 2)
    this._storage(offset) = value
    this
  }

  override def setValue(d1: Int, value: T): this.type = {
    require(1 == this.nDimension, "invalid size")
    var offset = this._storageOffset
    offset += getOffset(d1 - 1, 1)
    this._storage(offset) = value
    this
  }

  override def update(func: T => Boolean, value: T): Unit = {
    val func2 = new TensorFunc2[T] {
      override def apply(data: Array[T], index: Int): Unit = {
        data(index) = if (func(data(index))) value else data(index)
      }
    }
    DenseTensorApply.apply1[T](this, func2)
  }

  override def isContiguous(): Boolean = {
    DenseTensor.isContiguous(this)
  }

  override def contiguous(): Tensor[T] = {
    DenseTensor.newContiguous(this)
  }

  override def isSameSizeAs(other: Tensor[_]): Boolean = {
    DenseTensor.isSameSizeAs(this, other)
  }

  override def split(size: Int, dim: Int): Array[Tensor[T]] = {
    val result = new ArrayBuffer[Tensor[T]]()
    val dimLength = this.size(dim)
    var start = 1
    while (start <= dimLength) {
      val curSize = math.min(size, dimLength - start + 1)
      result.append(this.narrow(dim, start, curSize))
      start += curSize
    }
    result.toArray
  }

  override def split(dim: Int): Array[Tensor[T]] = {
    val result = new ArrayBuffer[Tensor[T]]()
    val dimLength = this.size(dim)
    var start = 1
    while (start <= dimLength) {
      result.append(this.select(dim, start))
      start += 1
    }
    result.toArray
  }

  // scalastyle:off methodName
  override def +(s: T): Tensor[T] = DenseTensorMath.add(s, this)

  override def +(t: Tensor[T]): Tensor[T] = DenseTensorMath.add(this, t)

  override def -(s: T): Tensor[T] = DenseTensorMath.sub(s, this)

  override def -(t: Tensor[T]): Tensor[T] = DenseTensorMath.sub(this, t)

  override def unary_-(): Tensor[T] = DenseTensorMath.neg(this)

  override def /(s: T): Tensor[T] = DenseTensorMath.divide(s, this)

  override def /(t: Tensor[T]): Tensor[T] = DenseTensorMath.divide(this, t)

  override def *(s: T): Tensor[T] = DenseTensorMath.mul(s, this)

  override def *(t: Tensor[T]): Tensor[T] = DenseTensorMath.mul(this, t)

  // scalastyle:on methodName

  override def sum(): T = DenseTensorMath.sumAll(this)

  override def sum(dim: Int): Tensor[T] = DenseTensorMath.sum(null, this, dim - 1)

  override def sum(x: Tensor[T], dim: Int): Tensor[T] = DenseTensorMath.sum(this, x, dim - 1)

  override def mean(): T = DenseTensorMath.meanAll(this)

  override def mean(dim: Int): Tensor[T] = DenseTensorMath.mean(this, dim - 1)

  override def max(): T = DenseTensorMath.maxAll(this)

  override def max(dim: Int): (Tensor[T], Tensor[T]) = {
    require(dim > 0 && dim <= this.nDimension, "dimension out of range")
    max(Tensor[T](), Tensor[T](), dim)
  }

  override def max(values: Tensor[T], indices: Tensor[T], dim: Int): (Tensor[T], Tensor[T]) = {
    require(dim > 0 && dim <= this.nDimension, "dimension out of range")
    val sizes = this.size() // here slice
    sizes(dim - 1) = 1
    values.resize(sizes)
    indices.resize(sizes)
    // TODO: the performance of contiguous tensor should be optimize
    DenseTensorDimApply.dimApply3[T](this, values, indices, dim, (tdata, toffset, tstride,
      tsize, vdata, voffset, vstride, vsize, idata, ioffset, istride, isize) => {
      var max = tdata(toffset)
      var index = 1
      var i = 0
      while (i < tsize) {
        if (ev.toType[Double](ev.minus(tdata(toffset + i * tstride), max)) > 0) {
          index = i + 1
          max = tdata(toffset + i * tstride)
        }
        i += 1
      }
      vdata(voffset) = max
      idata(ioffset) = ev.fromType[Float](index)
    })

    (values, indices)
  }

  override def min(): T = DenseTensorMath.minAll(this)

  override def min(dim: Int): (Tensor[T], Tensor[T]) = {
    require(dim > 0 && dim <= this.nDimension, "dimension out of range")
    min(Tensor[T](), Tensor[T](), dim)
  }

  override def min(values: Tensor[T], indices: Tensor[T], dim: Int): (Tensor[T], Tensor[T]) = {
    require(dim > 0 && dim <= this.nDimension, "dimension out of range")
    val sizes = this.size()
    sizes(dim - 1) = 1
    values.resize(sizes)
    indices.resize(sizes)
    // TODO: the performance of contiguous tensor should be optimize
    DenseTensorDimApply.dimApply3[T](this, values, indices, dim, (tdata, toffset, tstride,
      tsize, vdata, voffset, vstride, vsize, idata, ioffset, istride, isize) => {
      var min = tdata(toffset)
      var index = 1
      var i = 0
      while (i < tsize) {
        if (ev.isGreater(min, tdata(toffset + i * tstride))) {
          index = i + 1
          min = tdata(toffset + i * tstride)
        }
        i += 1
      }
      vdata(voffset) = min
      idata(ioffset) = ev.fromType[Float](index)
    })

    (values, indices)
  }

  def scatter(dim: Int, index: Tensor[T], src: Tensor[T]): Tensor[T] = {
    require(src.dim() == this.dim(), "Input tensor must have same dimensions as output tensor")
    require(dim <= this.dim(), "Index dimension is out of bounds")
    require(index.dim() == src.dim(), "Index tensor must have same dimensions as input tensor")
    val elementsPerRow = index.size(dim)
    // TODO: the performance of contiguous tensor should be optimize
    DenseTensorDimApply.dimApply3[T](this, src, index, dim, (tdata, toffset, tstride,
      tsize, vdata, voffset, vstride, vsize, idata, ioffset, istride, isize) => {
      var i = 0
      while (i < elementsPerRow) {
        val idx = ev.toType[Int](idata(ioffset + i * istride))
        require(idx >= 1 && idx <= this.size(dim))
        tdata((idx - 1) * tstride + toffset) = vdata(i * vstride + voffset)
        i += 1
      }
    })

    this
  }

  def gather(dim: Int, index: Tensor[T], src: Tensor[T]): Tensor[T] = {
    require(src.dim() == this.dim(), "Input tensor must have same dimensions as output tensor")
    require(dim <= this.dim(), "Index dimension is out of bounds")
    require(index.dim() == src.dim(), "Index tensor must have same dimensions as input tensor")
    val elementsPerRow = index.size(dim)
    // TODO: the performance of contiguous tensor should be optimize
    DenseTensorDimApply.dimApply3[T](this, src, index, dim, (tdata, toffset, tstride,
       tsize, vdata, voffset, vstride, vsize, idata, ioffset, istride, isize) => {
      var i = 0
      while (i < elementsPerRow) {
        val idx = ev.toType[Int](idata(ioffset + i * istride))
        require(idx >= 1 && idx <= src.size(dim), "invalid index in gather")
        tdata(i * tstride + toffset) = vdata((idx - 1) * vstride + voffset)
        i += 1
      }
    })

    this
  }

  override def add(value: T, y: Tensor[T]): Tensor[T] = DenseTensorMath.cadd(this, this, value, y)

  override def add(x: Tensor[T]): Tensor[T] = {
    require(this.nElement() == x.nElement())
    if (MKL.isMKLLoaded && this.isContiguous() && x.isContiguous()) {
      ev.vAdd(this.nElement(), this.storage().array(), this.storageOffset() - 1,
        x.storage().array(), x.storageOffset() - 1,
        this.storage().array(), this.storageOffset() - 1)
    }
    else {
      val func = new TensorFunc4[T] {
        override def apply (data1: Array[T], offset1: Int, data2: Array[T], offset2: Int): Unit = {
          data1(offset1) = ev.plus(data1(offset1), data2(offset2))
        }
      }
      DenseTensorApply.apply2[T](this, x, func)
    }
    this
  }

  override def add(x: Tensor[T], y: Tensor[T]): Tensor[T] = {
    require(this.nElement() == x.nElement() && this.nElement() == y.nElement())
    if (MKL.isMKLLoaded && this.isContiguous() && x.isContiguous() && y.isContiguous()) {
      ev.vAdd(this.nElement(), y.storage().array(), y.storageOffset() - 1,
        x.storage().array(), x.storageOffset() - 1,
        this.storage().array(), this.storageOffset() - 1)
    } else {
      val func = new TensorFunc6[T] {
        override def apply (data: Array[T], offset: Int, data1: Array[T],
                           offset1: Int, data2: Array[T], offset2: Int): Unit = {
          data(offset1) = ev.plus(data1(offset1), data2(offset2))
        }
      }
      DenseTensorApply.apply3[T](this, x, y, func)
    }
    this
  }

  // Puts the result of x + value * y in current tensor
  override def add(x: Tensor[T], value: T, y: Tensor[T]): Tensor[T] =
    DenseTensorMath.cadd(this, x, value, y)

  override def add(value: T): Tensor[T] = {
    if (this.isContiguous()) {
      ev.add(this.nElement(), this.storage().array(), this.storageOffset() - 1, value, 1)
      this
    } else {
      this.apply1(ev.plus(_, value))
    }
  }

  override def sub(value: T, y: Tensor[T]): Tensor[T] =
    DenseTensorMath.csub(this, this, ev.negative(value), y)

  override def sub(x: Tensor[T]): Tensor[T] = {
    require(this.nElement() == x.nElement())
    if (MKL.isMKLLoaded && this.isContiguous() && x.isContiguous()) {
      ev.vSub(this.nElement(), this.storage().array(), this.storageOffset() - 1,
        x.storage().array(), x.storageOffset() - 1,
        this.storage().array(), this.storageOffset() - 1)
    }
    else {
      val func = new TensorFunc4[T] {
        override def apply (data1: Array[T], offset1: Int, data2: Array[T], offset2: Int): Unit = {
          data1(offset1) = ev.minus(data1(offset1), data2(offset2))
        }
      }
      DenseTensorApply.apply2[T](this, x, func)
    }
    this
  }

  override def sub(x: Tensor[T], y: Tensor[T]): Tensor[T] = {
    require(this.nElement() == x.nElement() && this.nElement() == y.nElement())
    if (MKL.isMKLLoaded && this.isContiguous() && x.isContiguous() && y.isContiguous()) {
      ev.vSub(this.nElement(), x.storage().array(), x.storageOffset() - 1,
        y.storage().array(), y.storageOffset() - 1,
        this.storage().array(), this.storageOffset() - 1)
    } else {
      val func = new TensorFunc6[T] {
        override def apply (data: Array[T], offset: Int, data1: Array[T],
                           offset1: Int, data2: Array[T], offset2: Int): Unit = {
          data(offset1) = ev.minus(data1(offset1), data2(offset2))
        }
      }
      DenseTensorApply.apply3[T](this, x, y, func)
    }
    this
  }
  // Puts the result of x - value * y in current tensor
  override def sub(x: Tensor[T], value: T, y: Tensor[T]): Tensor[T] =
    DenseTensorMath.csub(this, x, value, y)

  override def sub(value: T): Tensor[T] = {
    if (this.isContiguous()) {
      ev.sub(this.nElement(), this.storage().array(), this.storageOffset() - 1, value, 1)
      this
    } else {
      this.apply1(ev.minus(_, value))
    }
  }

  override def dot(y: Tensor[T]): T = {
    var sum = ev.fromType[Int](0)
    this.map(y, (a, b) => {
      sum = ev.plus(sum, ev.times(a, b))
      a
    })
    sum
  }

  override def cmax(value: T): Tensor[T] = {
    this.apply1(x => ev.max(x, value))
  }

  override def dist(y: Tensor[T], norm: Int): T = {
    var sum = ev.fromType[Int](0)
    this.map(y, (a, b) => {
      sum = ev.plus(sum, ev.pow(ev.abs(ev.minus(b, a)), ev.fromType[Int](norm)))
      a
    })
    ev.pow(sum, ev.divide(ev.fromType[Int](1), ev.fromType[Int](norm)))
  }

  override def addcmul(value: T, tensor1: Tensor[T], tensor2: Tensor[T]): Tensor[T] = {
    require(tensor1.nElement() == tensor2.nElement() && this.nElement() == tensor1.nElement())

    if (this.isContiguous() && tensor1.isContiguous() && tensor2.isContiguous()) {
      ev.getType() match {
        case DoubleType =>
          val v = value.asInstanceOf[Double]
          val t1 = tensor1.storage().array().asInstanceOf[Array[Double]]
          val t1Offset = tensor1.storageOffset() - 1
          val t2 = tensor2.storage().array().asInstanceOf[Array[Double]]
          val t2Offset = tensor2.storageOffset() - 1
          val self = this.storage().array().asInstanceOf[Array[Double]]
          val selfOffset = this.storageOffset() - 1
          val n = this.nElement()
          var i = 0

          while (i < n) {
            self(i + selfOffset) += t1(t1Offset + i) * t2(t2Offset + i) * v
            i += 1
          }
        case FloatType =>
          val v = value.asInstanceOf[Float]
          val t1 = tensor1.storage().array().asInstanceOf[Array[Float]]
          val t1Offset = tensor1.storageOffset() - 1
          val t2 = tensor2.storage().array().asInstanceOf[Array[Float]]
          val t2Offset = tensor2.storageOffset() - 1
          val self = this.storage().array().asInstanceOf[Array[Float]]
          val selfOffset = this.storageOffset() - 1
          val n = this.nElement()
          var i = 0
          while (i < n) {
            self(i + selfOffset) += t1(t1Offset + i) * t2(t2Offset + i) * v
            i += 1
          }
      }
    } else {
      val func = new TensorFunc6[T] {
        override def apply(data1: Array[T], offset1: Int, data2: Array[T], offset2: Int,
          data3: Array[T], offset3: Int): Unit = {
          data1(offset1) = ev.plus(data1(offset1), ev.times(ev.times(data2(offset2),
            data3(offset3)), value))
        }
      }
      DenseTensorApply.apply3[T](this, tensor1, tensor2, func)
    }
    this
  }

  override def addcmul(tensor1: Tensor[T], tensor2: Tensor[T]): Tensor[T] =
    addcmul(ev.fromType(1), tensor1, tensor2)

  override def addcdiv(value: T, tensor1: Tensor[T], tensor2: Tensor[T]): Tensor[T] = {
    if (this.isContiguous() && tensor1.isContiguous() && tensor2.isContiguous()) {
      ev.getType() match {
        case DoubleType =>
          val v = value.asInstanceOf[Double]
          val t1 = tensor1.storage().array().asInstanceOf[Array[Double]]
          val t1Offset = tensor1.storageOffset() - 1
          val t2 = tensor2.storage().array().asInstanceOf[Array[Double]]
          val t2Offset = tensor2.storageOffset() - 1
          val self = this.storage().array().asInstanceOf[Array[Double]]
          val selfOffset = this.storageOffset() - 1
          val n = this.nElement()
          var i = 0

          while (i < n) {
            self(i + selfOffset) += t1(t1Offset + i) / t2(t2Offset + i) * v
            i += 1
          }
        case FloatType =>
          val v = value.asInstanceOf[Float]
          val t1 = tensor1.storage().array().asInstanceOf[Array[Float]]
          val t1Offset = tensor1.storageOffset() - 1
          val t2 = tensor2.storage().array().asInstanceOf[Array[Float]]
          val t2Offset = tensor2.storageOffset() - 1
          val self = this.storage().array().asInstanceOf[Array[Float]]
          val selfOffset = this.storageOffset() - 1
          val n = this.nElement()
          var i = 0

          while (i < n) {
            self(i + selfOffset) += t1(t1Offset + i) / t2(t2Offset + i) * v
            i += 1
          }
      }
    } else {
      val func = new TensorFunc6[T] {
        override def apply(data1: Array[T], offset1: Int, data2: Array[T], offset2: Int,
          data3: Array[T], offset3: Int): Unit = {
          data1(offset1) = ev.plus(data1(offset1), ev.times(ev.divide(data2(offset2),
            data3(offset3)), value))
        }
      }
      DenseTensorApply.apply3[T](this, tensor1, tensor2, func)
    }
    this
  }

  override def cmul(y: Tensor[T]): Tensor[T] = DenseTensorMath.cmul(this, this, y)

  override def cmul(x: Tensor[T], y: Tensor[T]): Tensor[T] = DenseTensorMath.cmul(this, x, y)

  override def cdiv(y: Tensor[T]): Tensor[T] = DenseTensorMath.cdiv(this, this, y)

  override def cdiv(x: Tensor[T], y: Tensor[T]): Tensor[T] = DenseTensorMath.cdiv(this, x, y)

  /**
   * stores the element-wise maximum of x and y in x.
   * x.cmax(y) = max(x, y)
   *
   * @param y tensor
   * @return current tensor
   */
  override def cmax(y: Tensor[T]): Tensor[T] = DenseTensorMath.cmax(this, this, y)

  /**
   * stores the element-wise maximum of x and y in z.
   * z.cmax(x, y) means z = max(x, y)
   *
   * @param x tensor
   * @param y tensor
   */
  override def cmax(x: Tensor[T], y: Tensor[T]): Tensor[T] = DenseTensorMath.cmax(this, x, y)

  override def mul(x: Tensor[T], value: T): Tensor[T] = DenseTensorMath.mul(this, x, value)

  override def mul(value: T): Tensor[T] = DenseTensorMath.mul(this, null, value)

  override def div(value: T): Tensor[T] = DenseTensorMath.mul(this, null, ev.inv(value))

  override def conv2(kernel: Tensor[T], vf: Char = 'V'): Tensor[T] =
    DenseTensorConv.conv2Dmul[T](ev.fromType[Int](1), this, kernel, 1, 1, vf, 'C')

  override def xcorr2(kernel: Tensor[T], vf: Char = 'V'): Tensor[T] =
    DenseTensorConv.conv2Dmul[T](ev.fromType[Int](1), this, kernel, 1, 1, vf, 'X')

  override def addmm(v1: T, M: Tensor[T], v2: T, mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] =
    DenseTensorMath.addmm(this, v1, M, v2, mat1, mat2)

  override def addmm(M: Tensor[T], mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] =
    DenseTensorMath.addmm[T](this, ev.fromType[Int](1), M, ev.fromType[Int](1), mat1, mat2)

  override def addmm(mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] =
    DenseTensorMath.addmm[T](this, ev.fromType[Int](1), this, ev.fromType[Int](1), mat1, mat2)

  override def addmm(v2: T, mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] =
    DenseTensorMath.addmm[T](this, ev.fromType[Int](1), this, v2, mat1, mat2)

  override def addmm(v1: T, v2: T, mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] =
    DenseTensorMath.addmm(this, v1, this, v2, mat1, mat2)

  override def mm(mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] =
    DenseTensorMath.addmm(this, ev.fromType[Int](1), this, ev.fromType[Int](1), mat1, mat2)

  override def addr(t1: Tensor[T], t2: Tensor[T]): Tensor[T] =
    DenseTensorMath.addr[T](this, ev.fromType[Int](1), this, ev.fromType[Int](1), t1, t2)

  override def addr(v2: T, t1: Tensor[T], t2: Tensor[T]): Tensor[T] =
    DenseTensorMath.addr[T](this, ev.fromType[Int](1), this, v2, t1, t2)

  override def addr(v1: T, t1: Tensor[T], v2: T, t2: Tensor[T]): Tensor[T] =
    DenseTensorMath.addr(this, v1, this, v2, t1, t2)

  /**
   * Performs the outer-product between vec1 (1D Tensor) and vec2 (1D Tensor).
   * Optional values v1 and v2 are scalars that multiply mat and vec1 [out] vec2 respectively.
   * In other words,res_ij = (v1 * mat_ij) + (v2 * vec1_i * vec2_j)
   *
   * @param v1
   * @param t1
   * @param v2
   * @param t2
   * @param t3
   * @return
   */
  override def addr(v1: T, t1: Tensor[T], v2: T, t2: Tensor[T], t3: Tensor[T]): Tensor[T] =
    DenseTensorMath.addr(this, v1, t1, v2, t2, t3)

  override def addmv(beta: T, vec1: Tensor[T], alpha: T, mat: Tensor[T],
    vec2: Tensor[T]): Tensor[T] =
    DenseTensorMath.addmv(this, beta, vec1, alpha, mat, vec2)

  override def addmv(beta: T, alpha: T, mat: Tensor[T], vec2: Tensor[T]): Tensor[T] =
    DenseTensorMath.addmv(this, beta, this, alpha, mat, vec2)

  /**
   * return pseudo-random numbers, require 0<=args.length<=2
   * if args.length = 0, return [0, 1)
   * if args.length = 1, return [1, args(0)] or [args(0), 1]
   * if args.length = 2, return [args(0), args(1)]
   *
   * @param args
   */
  override def uniform(args: T*): T = {
    require(args.length <= 2, s"invalid arguments, excepted ${args.length} <= 2.")
    if (args.length == 0) {
      ev.rand()
    } else if (args.length == 1) {
      ev.plus(ev.times(ev.rand(), ev.minus(args(0), ev.fromType[Int](1))),
        ev.fromType[Int](1))
    } else {
      require(ev.toType[Double](ev.minus(args(0), args(1))) <= 0.0,
        s"invalid arguments, excepted ${args(0)} <= ${args(1)}.")
      ev.plus(ev.times(ev.rand(), ev.minus(args(1), args(0))), args(0))
    }
  }

  override def repeatTensor(sizes: Array[Int]): Tensor[T] = {
    require(sizes.length >= this.nDimension,
      "Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor")
    val result = new DenseTensor[T]()
    val xTensor = this.clone()
    var xSize = xTensor.size()
    var i = 1
    while (i <= sizes.length - this.dim()) {
      xSize = Array(1) ++ xSize
      i += 1
    }
    val size = new DenseTensor(Storage[T](xSize.map(x => ev.fromType[Int](x)))).
      cmul(new DenseTensor(Storage[T](sizes.map(x => ev.fromType[Int](x))))).
      storage().array().map(x => ev.toType[Int](x))
    xTensor.resize(xSize)
    result.resize(size)
    var urTensor = Tensor(result)

    i = 1
    while (i <= xTensor.dim()) {
      urTensor = urTensor.unfold(i, xTensor.size(i), xTensor.size(i))
      i += 1
    }

    i = 1
    while (i <= urTensor.dim() - xTensor.dim()) {
      xSize = Array(1) ++ xSize
      i += 1
    }

    xTensor.resize(xSize)
    val xxTensor = xTensor.expandAs(urTensor)
    urTensor.copy(xxTensor)
    result
  }

  override def expandAs(template: Tensor[T]): Tensor[T] = {
    this.expand(template.size())
  }

  override def expand(sizes: Array[Int]): Tensor[T] = {
    require(sizes.length == this.dim(),
      s"the number of dimensions provided must equal ${this.dim()}")
    val tensorDim = this.dim()
    val tensorStride = this.stride()
    val tensorSize = this.size()

    var i = 0
    while (i < tensorDim) {
      if (tensorSize(i) == 1) {
        tensorSize(i) = sizes(i)
        tensorStride(i) = 0
      } else if (tensorSize(i) != sizes(i)) {
        throw new UnsupportedOperationException(
          "incorrect size: only supporting singleton expansion (size=1)")
      }
      i += 1
    }

    set(this.storage(), this.storageOffset(), tensorSize, tensorStride)
  }

  override def addmv(alpha: T, mat: Tensor[T], vec2: Tensor[T]): Tensor[T] =
    DenseTensorMath.addmv(this, ev.fromType[Int](1), this, alpha, mat, vec2)

  override def mv(mat: Tensor[T], vec2: Tensor[T]): Tensor[T] =
    DenseTensorMath.addmv(this, ev.fromType[Int](1), this, ev.fromType[Int](1), mat, vec2)

  override def baddbmm(beta: T, M: Tensor[T], alpha: T, batch1: Tensor[T],
    batch2: Tensor[T]): Tensor[T] = DenseTensorMath.baddbmm(this, beta, M, alpha, batch1, batch2)

  override def baddbmm(beta: T, alpha: T, batch1: Tensor[T], batch2: Tensor[T]): Tensor[T] =
    DenseTensorMath.baddbmm(this, beta, this, alpha, batch1, batch2)

  override def baddbmm(alpha: T, batch1: Tensor[T], batch2: Tensor[T]): Tensor[T] =
    DenseTensorMath.baddbmm(this, ev.fromType[Int](1), this, alpha, batch1, batch2)

  override def bmm(batch1: Tensor[T], batch2: Tensor[T]): Tensor[T] =
    DenseTensorMath.baddbmm(this, ev.fromType[Int](1), this, ev.fromType[Int](1), batch1, batch2)

  override def abs(): Tensor[T] = this.apply1(ev.abs(_))

  override def toBreezeVector(): BrzDenseVector[T] = {
    require(this.nDimension == 1, "tensor is not 1D")
    new BrzDenseVector(this.storage().array(), this.storageOffset() - 1, this.stride(1),
      this.nElement())
  }

  override def getType(): TensorDataType = ev.getType()

  override def toMLlibMatrix(): Matrix = {
    require(this.nDimension == 2, "tensor is not 2D")
    require((this.stride(1) == 1 && this.stride(2) == this.size(1))
      || (this.stride(1) == this.size(2) && this.stride(2) == 1), "tensor is not continuous")
    new DenseMatrix(this.size(1), this.size(2), this.storage().array().asInstanceOf[Array[Double]],
      this.stride(2) == 1) // column major
  }

  override def toBreezeMatrix(): BrzDenseMatrix[T] = {
    require(this.nDimension == 2, "tensor is not 2D")
    val majorStride = if (this.stride(2) == 1) this.stride(1) else this.stride(2)
    new BrzDenseMatrix[T](this.size(1), this.size(2), this.storage().array(),
      this.storageOffset() - 1,
      majorStride, this.stride(2) == 1)
  }

  override def toMLlibVector(): Vector = {
    require(this.nDimension == 1, "tensor is not 1D")
    require(this.stride(1) == 1, "tensor is not continuous")
    new DenseVector(this.storage().array().asInstanceOf[Array[Double]])
  }

  override def equals(obj: Any): Boolean = {
    if (obj == null) {
      return false
    }
    if (!obj.isInstanceOf[Tensor[T]]) {
      return false
    }
    val other = obj.asInstanceOf[Tensor[T]]
    if (this.eq(other)) {
      return true
    }
    if (this.nDimension != other.nDimension) {
      return false
    }
    var d = 1
    while (d <= this.nDimension) {
      if (this.size(d) != other.size(d)) {
        return false
      }
      d += 1
    }
    var result = true
    this.map(other, (a, b) => {
      if (result) {
        ev.getType() match {
          case FloatType =>
            result = DenseTensorMath.nearlyEqual(a, b, DenseTensorMath.floatEpsilon)
          case DoubleType =>
            result = DenseTensorMath.nearlyEqual(a, b, DenseTensorMath.doubleEpsilon)
        }
      }
      a
    })
    return result
  }

  override def hashCode(): Int = {
    val seed = 37
    var hash = 1
    hash = hash * seed + this.nDimension
    var d = 1
    while (d <= this.nDimension) {
      hash = hash * seed + this.size(d)
      d += 1
    }
    this.apply1(e => {
      hash = hash * seed + e.hashCode()
      e
    })

    hash
  }

  override def toString(): String = {
    this.nDimension match {
      case 0 => s"[${this.getClass.getName} with no dimension]"
      case 1 =>
        val sb = new StringBuilder
        this.apply1(e => {
          sb.append(e).append('\n')
          e
        })

        s"${sb}[${this.getClass.getName} of size ${this.size(1)}]"
      case 2 =>
        val sb = new StringBuilder
        val indexer = Array(0, 0)
        var i = 1
        while (i <= this.size(1)) {
          var j = 1
          while (j <= this.size(2)) {
            indexer(0) = i
            indexer(1) = j
            sb.append(this.apply(indexer)).append('\t')
            j += 1
          }
          sb.append('\n')
          i += 1
        }

        s"${sb}[${this.getClass.getName} of size ${this.size(1)}x${this.size(2)}]"
      case _ =>
        val sb = new StringBuilder
        val size = this.size()
        val indexer = Array.fill(this.nDimension)(1)
        var done = false
        val _lastDim = this.nDimension - 1
        val _secLastDim = _lastDim - 1
        var d = _secLastDim - 1
        val total = this.nElement()
        while (!done) {
          // print header
          sb.append('(')
          var i = 0
          while (i < _secLastDim) {
            sb.append(indexer(i)).append(',')
            i += 1
          }
          sb.append(".,.) =\n")

          // print current matrix
          i = 1
          while (i <= this.size(_secLastDim + 1)) {
            var j = 1
            while (j <= this.size(_lastDim + 1)) {
              indexer(_lastDim) = j
              indexer(_secLastDim) = i
              sb.append(this.apply(indexer)).append('\t')
              j += 1
            }
            sb.append('\n')
            i += 1
          }

          sb.append('\n')
          indexer(d) = indexer(d) + 1
          while (d >= 0 && indexer(d) > size(d)) {
            indexer(d) = 1
            d = d - 1
            if (d >= 0) indexer(d) = indexer(d) + 1
          }

          if (d == -1) {
            done = true
          } else {
            d = _secLastDim - 1
          }
        }
        s"${sb}[${this.getClass.getName} of size ${size.mkString("x")}]"
    }
  }

  override def diff(other: Tensor[T], count: Int, reverse: Boolean): Boolean = {
    if (this.nDimension != other.nDimension()) {
      println("Dimension number is different")
      return true
    }

    var d = 1
    while (d <= this.nDimension) {
      if (this.size(d) != other.size(d)) {
        println(s"Dimension $d is different, left is ${this.size(d)}, right is ${other.size(d)}")
        return true
      }
      d += 1
    }

    val buffer = new Array[(T, T, Int)](count)
    var result = false
    var catchNum = 0
    val func2 = new TensorFunc4[T] {
      override def apply(data1: Array[T], offset1: Int, data2: Array[T], offset2: Int): Unit = {
        if (data1(offset1) != data2(offset2)) {
          require(offset1 == offset2)
          if (reverse || catchNum < count) {
            buffer(catchNum % count) = (data1(offset1), data2(offset2), offset1)
          }
          catchNum += 1
          result = true
        }
      }
    }
    DenseTensorApply.apply2[T](this, other, func2)

    if (result == true) {
      var i = 0
      while (i < buffer.length) {
        println(
          s"Find difference => this is ${buffer(i)._1} other is ${buffer(i)._2} " +
            s"offset is (${buffer(i)._3}/${this.nElement()}})")
        i += 1
      }
    }

    result
  }

  override def reshape(sizes: Array[Int]): Tensor[T] = {
    require(sizes.product == this.nElement(),
      "DenseTensor: nElement of this tensor is not equal to nElement specified by sizes," +
        s" specified sizes = (${sizes.mkString(",")})," +
        s" nElement specified by sizes = ${sizes.reduce(_ * _)}," +
        s" nElement of this tensor = ${this.nElement()}")
    val result = new DenseTensor[T]()
    result.resize(sizes)
    result.copy(this)
    result
  }

  override def topk(k: Int, dim: Int, increase: Boolean, result: Tensor[T],
    indices: Tensor[T]): (Tensor[T], Tensor[T]) = {
    val selectDim = if (dim == -1) this.dim() else dim
    require(selectDim > 0 && selectDim <= this.nDimension)

    val sliceSize = this.size(selectDim)
    require(k > 0 && k <= sliceSize)

    val tmpResult = new Array[(T, Int)](sliceSize)

    val topKSize = this.size()
    topKSize(selectDim - 1) = k

    val resultTensor = if (result == null) Tensor[T]() else result
    resultTensor.resize(topKSize)

    val indicesTensor = if (indices == null) Tensor[T]() else indices
    indicesTensor.resize(topKSize)

    DenseTensorDimApply.dimApply3[T](this, resultTensor, indicesTensor, selectDim,
      (tdata, toffset, tstride, tsize, vdata, voffset, vstride, vsize, idata,
      ioffset, istride, isize) => {
        var i = 0
        while (i < tsize) {
          tmpResult(i) = (tdata(toffset + i * tstride), i + 1)
          i += 1
        }
        val sorted = tmpResult.sortWith((l, r) =>
          if (increase) ev.isGreater(r._1, l._1) else ev.isGreater(l._1, r._1))
        i = 0
        while (i < k) {
          vdata(voffset + i * vstride) = sorted(i)._1
          idata(ioffset + i * istride) = ev.fromType(sorted(i)._2)
          i += 1
        }
      })

    (resultTensor, indicesTensor)
  }

  override def pow(x: Tensor[T], n: T): Tensor[T] = DenseTensorMath.pow[T](this, x, n)

  override def pow(n: T): Tensor[T] = DenseTensorMath.pow[T](this, this, n)

  override def log(x: Tensor[T]): Tensor[T] = DenseTensorMath.log[T](this, x)

  override def log(): Tensor[T] = DenseTensorMath.log[T](this, this)

  override def exp(x: Tensor[T]): Tensor[T] = DenseTensorMath.exp[T](this, x)

  override def exp(): Tensor[T] = DenseTensorMath.exp[T](this, this)

  override def sqrt(x: Tensor[T]): Tensor[T] = DenseTensorMath.sqrt[T](this, x)

  override def sqrt(): Tensor[T] = DenseTensorMath.sqrt[T](this, this)

  override def log1p(x: Tensor[T]): Tensor[T] = DenseTensorMath.log1p[T](this, x)

  override def log1p(): Tensor[T] = DenseTensorMath.log1p[T](this, this)

  override def norm(y: Tensor[T], value: Int, dim: Int): Tensor[T] =
    DenseTensorMath.norm(this, y, value, dim - 1)

  override def abs(x: Tensor[T]): Tensor[T] = {
    require(this.nElement() == x.nElement())
    if (MKL.isMKLLoaded && this.isContiguous() && x.isContiguous()) {
      ev.vAbs(this.nElement(), x.storage().array(), x.storageOffset() - 1,
        this.storage().array(), this.storageOffset() - 1)
    } else {
      val func = new TensorFunc4[T] {
        override def apply(data1: Array[T], offset1: Int, data2: Array[T], offset2: Int): Unit = {
          data1(offset1) = ev.abs(data2(offset2))
        }
      }
      DenseTensorApply.apply2[T](this, x, func)
    }
    this
  }

  override def save(path: String, overWrite: Boolean): this.type = {
    File.save(this, path, overWrite)
    this
  }

  /**
   * Fills the masked elements of itself with value val
   *
   * @param mask
   * @param value
   * @return current tensor reference
   */
  override def maskedFill(mask: Tensor[T], value: T): Tensor[T] = {
    require(this.nElement() == mask.nElement())

    // todo: the performance of contiguous tensor should be optimized
    val func = new TensorFunc4[T] {
      def apply(data1: Array[T], offset1: Int, data2: Array[T], offset2: Int): Unit = {
        require(ev.toType[Int](data2(offset2)) == 1 || ev.toType[Int](data2(offset2)) == 0,
          "Mask tensor can take 0 and 1 values only")
        if (ev.toType[Int](data2(offset2)) == 1) {
          data1(offset1) = value
        }
      }
    }
    DenseTensorApply.apply2[T](this, mask, func)
    this
  }

  /**
   * Copies the elements of tensor into mask locations of itself.
   *
   * @param mask
   * @param y
   * @return current tensor reference
   */
  override def maskedCopy(mask: Tensor[T], y: Tensor[T]): Tensor[T] = {
    require(this.nElement() == mask.nElement())
    require(y.isContiguous())

    val data3 = y.storage().array()
    var offset = 0
    // todo: the performance of contiguous tensor should be optimized
    val func = new TensorFunc4[T] {
      override def apply(data1: Array[T], offset1: Int, data2: Array[T], offset2: Int): Unit = {
        require(ev.toType[Int](data2(offset2)) == 1 || ev.toType[Int](data2(offset2)) == 0,
          "Mask tensor can take 0 and 1 values only")
        if (ev.toType[Int](data2(offset2)) == 1) {
          require(offset < data3.length, "Number of elements of y < number of ones in mask")
          data1(offset1) = data3(offset)
          offset += 1
        }
      }
    }
    DenseTensorApply.apply2[T](this, mask, func)
    this
  }

  /**
   * Returns a new Tensor which contains all elements aligned to a 1 in the corresponding mask.
   *
   * @param mask
   * @param res
   * @return current tensor reference
   */
  override def maskedSelect(mask: Tensor[T], res: Tensor[T]): Tensor[T] = {
    require(this.nElement() == mask.nElement())
    require(ev.isGreater(mask.sum(), ev.fromType(0)))
    val length = mask.sum()
    var offset = 0
    res.resize(ev.toType[Double](length).toInt)
    val result = res.storage().array()

    // todo: the performance of contiguous tensor should be optimized
    val func = new TensorFunc4[T] {
      override def apply(data1: Array[T], offset1: Int, data2: Array[T], offset2: Int): Unit = {
        require(ev.toType[Int](data2(offset2)) == 1 || ev.toType[Int](data2(offset2)) == 0,
          "Mask tensor can take 0 and 1 values only")
        if (ev.toType[Int](data2(offset2)) == 1) {
          result(offset) = data1(offset1)
          offset += 1
        }
      }
    }
    DenseTensorApply.apply2[T](this, mask, func)
    res
  }

  /**
   * Implements > operator comparing each element in x with y
   *
   * @param x
   * @param y
   * @return current tensor reference
   */
  override def gt(x: Tensor[T], y: Tensor[T]): Tensor[T] = {
    // todo: the performance of contiguous tensor should be optimized
    val func = new TensorFunc6[T] {
      def apply(data1: Array[T], offset1: Int, data2: Array[T], offset2: Int,
                data3: Array[T], offset3: Int): Unit = {
        if (ev.isGreater(data2(offset1), data3(offset2))) {
          data1(offset1) = ev.fromType(1)
        } else {
          data1(offset1) = ev.fromType(0)
        }
      }
    }
    DenseTensorApply.apply3[T](this, x, y, func)
    this
  }
  /**
   * Implements < operator comparing each element in x with y
   *
   * @param x
   * @param y
   * @return current tensor reference
   */
  override def lt(x: Tensor[T], y: Tensor[T]): Tensor[T] = {
    // todo: the performance of contiguous tensor should be optimized
    val func = new TensorFunc6[T] {
      def apply(data1: Array[T], offset1: Int, data2: Array[T], offset2: Int,
                data3: Array[T], offset3: Int): Unit = {
        if (ev.toType[Double](ev.minus(data2(offset1), data3(offset2))) < 0) {
          data1(offset1) = ev.fromType(1)
        } else {
          data1(offset1) = ev.fromType(0)
        }
      }
    }
    DenseTensorApply.apply3[T](this, x, y, func)
    this
  }

  /**
   * Implements <= operator comparing each element in x with y
   *
   * @param x
   * @param y
   * @return current tensor reference
   */
  override def le(x: Tensor[T], y: Tensor[T]): Tensor[T] = {
    // todo: the performance of contiguous tensor should be optimized
    val func = new TensorFunc6[T] {
      def apply(data1: Array[T], offset1: Int, data2: Array[T], offset2: Int,
                data3: Array[T], offset3: Int): Unit = {
        if (ev.toType[Double](ev.minus(data2(offset1), data3(offset2))) <= 0) {
          data1(offset1) = ev.fromType(1)
        } else {
          data1(offset1) = ev.fromType(0)
        }
      }
    }
    DenseTensorApply.apply3[T](this, x, y, func)
    this
  }

  /**
   * Implements == operator comparing each element in a with b
 *
   * @param x
   * @param value
   * @return
   */
  override def eq(x: Tensor[T], value: T): Tensor[T] = {
    // todo: the performance of contiguous tensor should be optimized
    val func = new TensorFunc4[T] {
      def apply(data1: Array[T], offset1: Int, data2: Array[T], offset2: Int): Unit = {
        if (data2(offset1) == value) {
          data1(offset1) = ev.fromType(1)
        } else {
          data1(offset1) = ev.fromType(0)
        }
      }
    }
    DenseTensorApply.apply2[T](this, x, func)
    this
  }

  /**
   * returns the sum of the n-norms on the Tensor x
   *
   * @param value the n-norms
   * @return
   */
  override def norm(value: Int): T = {
    require(value > 0, "norm value should be greater than 0")
    var res: T = ev.fromType(0)
    val func = new TensorFunc2[T] {
      override def apply(data1: Array[T], offset1: Int): Unit = {
        res = ev.plus(res, ev.pow(ev.abs(data1(offset1)), ev.fromType(value)))
      }
    }
    DenseTensorApply.apply1[T](this, func)
    ev.pow(res, ev.fromType(1.0 / value))
  }

  /**
   * returns a new Tensor with the sign (+/- 1 or 0) of the elements of x.
   *
   * @return
   */
  override def sign(): Tensor[T] = {
    val func = new TensorFunc2[T] {
      override def apply(data1: Array[T], offset1: Int): Unit = {
        if (ev.isGreater(data1(offset1), ev.fromType(0))) {
          data1(offset1) = ev.fromType(1)
        } else if (ev.isGreater(ev.fromType(0), data1(offset1))) {
          data1(offset1) = ev.fromType(-1)
        } else {
          data1(offset1) = ev.fromType(0)
        }
      }
    }
    DenseTensorApply.apply1[T](this, func)
    this
  }


  /**
   * resize this tensor size to floor((xmax - xmin) / step) + 1 and set values from
   * xmin to xmax with step (default to 1).
   * @param xmin
   * @param xmax
   * @param step
   * @return this tensor
   */
  override def range(xmin: Double, xmax: Double, step: Int = 1): Tensor[T] = {
    require((xmax >= xmin) && (step > 0),
      "upper bound and larger bound incoherent with step sign")
    val size = math.floor((xmax-xmin)/ step + 1).toInt
    if (this.nElement() != size) this.resize(size)
    var i = 0
    // TODO: the performance of contiguous tensor should be optimize
    val func = new TensorFunc2[T] {
      override def apply(data1: Array[T], offset1: Int): Unit = {
        data1(offset1) = ev.fromType(xmin + i * step)
        i += 1
      }
    }
    DenseTensorApply.apply1[T](this, func)
    this
  }

  override def addSingletonDimension(t: Tensor[T], dim: Int = 1): Tensor[T] = {
    require(dim > 0 && dim <= t.dim() + 1, s"invalid dimension: $dim. " +
      s"Tensor is of ${t.dim()} dimensions.")

    val size = new Array[Int](t.dim() + 1)
    val stride = new Array[Int](t.dim() + 1)

    var d = 0
    while (d < dim - 1) {
      size(d) = t.size(d + 1)
      stride(d) = t.stride(d + 1)
      d += 1
    }
    size(dim - 1) = 1
    stride(dim - 1) = 1
    d += 1
    while (d < t.dim + 1) {
      size(d) = t.size(d)
      stride(d) = t.stride(d)
      d += 1
    }

    this.set(t.storage(), t.storageOffset(), size, stride)
  }

  /**
   * Implements >= operator comparing each element in x with value
 *
   * @param x
   * @param value
   * @return
   */
  override def ge(x: Tensor[T], value: Double): Tensor[T] = {
    // todo: the performance of contiguous tensor should be optimized
    val func = new TensorFunc4[T] {
      def apply(data1: Array[T], offset1: Int, data2: Array[T], offset2: Int): Unit = {
        if (ev.toType[Double](data2(offset2)) >= value) {
          data1(offset1) = ev.fromType(1)
        } else {
          data1(offset1) = ev.fromType(0)
        }
      }
    }
    DenseTensorApply.apply2[T](this, x, func)
    this
  }

  /**
   * Accumulate the elements of tensor into the original tensor by adding to the indices
   * in the order given in index. The shape of tensor must exactly match the elements indexed
   * or an error will be thrown.
 *
   * @param dim
   * @param index
   * @param y
   * @return
   */
  override def indexAdd(dim: Int, index: Tensor[T], y: Tensor[T]): Tensor[T] = {
    require(dim <= y.nDimension(), "Indexing dim is out of bounds of tensor y")
    require(index.nElement() == y.size(dim),
      "Number of indices should be equal to source:size(dim)")
    require(index.nDimension() == 1, "Index is supposed to be a vector")

    val indexC = index.contiguous()
    val numEle = indexC.nElement()
    var i = 1
    if (this.nDimension > 1) {
      while (i <= numEle) {
        this.select(dim, ev.toType[Double](indexC(Array(i))).toInt).add(y.select(dim, i))
        i += 1
      }
    } else {
      while (i <= numEle) {
        this.narrow(1, ev.toType[Double](indexC(Array(i))).toInt, 1).add(y.narrow(1, i, 1))
        i += 1
      }
    }
    this
  }

  /**
   * create a new Tensor which indexes the original Tensor along dimension dim using the entries
   * in torch.LongTensor index. The returned Tensor has the same number of dimensions as the
   * original Tensor.
 *
   * @param dim
   * @param index
   * @param y
   * @return
   */
  override def index(dim: Int, index: Tensor[T], y: Tensor[T]): Tensor[T] = {
    require(dim <= y.nDimension(), "Indexing dim is out of bounds of tensor y")
    require(index.nDimension() == 1, "Index is supposed to be a vector")
    require(y.nDimension() > 0, "Source tensor is empty")
    val indexC = index.contiguous()

    val numEle = indexC.nElement()
    val newSize = y.size()
    newSize(dim - 1) = numEle
    this.resize(newSize)

    var i = 1
    if (y.nDimension() == 1) {
      while (i <= numEle) {
        this.narrow(1, i, 1).add(y.narrow(1, ev.toType[Double](indexC(Array(i))).toInt, 1))
        i += 1
      }
    } else {
      while (i <= numEle) {
        this.select(dim, i).copy(y.select(dim, ev.toType[Double](indexC(Array(i))).toInt))
        i += 1
      }
    }
    this
  }

  override def toTensor[D](implicit env: TensorNumeric[D]): Tensor[D] = {
    if (ev.getType() == env.getType()) {
      this.asInstanceOf[Tensor[D]]
    } else {
      throw new IllegalArgumentException(s"The type ${env.getType().getClass}" +
        s" in toTensor[${env.getType().getClass}] is not same" +
        s"as the numeric type ${ev.getType().getClass} of the " +
        "corresponding module, please keep them same.")
    }
  }
}

object DenseTensor {
  private[tensor] def squeeze[@specialized(Float, Double) T](self: DenseTensor[T]): Tensor[T] = {
    var ndim = 0
    var d = 0
    while (d < self.nDimension) {
      if (self._size(d) != 1) {
        if (d != ndim) {
          self._size(ndim) = self._size(d)
          self._stride(ndim) = self._stride(d)
        }
        ndim += 1
      }
      d += 1
    }

    if (ndim == 0 && self.nDimension > 0) {
      self._size(0) = 1
      self._stride(0) = 1
      ndim = 1
    }

    self.nDimension = ndim
    self
  }

  private[tensor] def squeeze[@specialized(Float, Double) T](self: DenseTensor[T],
    _dim: Int): Tensor[T] = {
    require(_dim >= 0 && _dim < self.nDimension, "dimension out of range")
    if (self._size(_dim) == 1 && self.nDimension > 1) {
      var d = _dim
      while (d < self.nDimension - 1) {
        self._size(d) = self._size(d + 1)
        self._stride(d) = self._stride(d + 1)
        d += 1
      }

      self.nDimension -= 1
    }
    self
  }

  private[tensor] def newWithStorage[@specialized(Float, Double) T: ClassTag](
    tensor: DenseTensor[T], storage: Storage[T], storageOffset: Int, size: Array[Int],
    stride: Array[Int], ev: TensorNumeric[T]): DenseTensor[T] = {
    if (size != null && stride != null) {
      require(size.length == stride.length, "inconsistent size")
    }

    implicit val ev2 = ev
    val self = if (tensor == null) new DenseTensor[T]() else tensor
    val nDimension = if (size != null) size.length else if (stride != null) stride.length else 0

    DenseTensor.rawSet[T](self, storage, storageOffset, nDimension, size, stride)
  }

  private[tensor] def newWithTensor[@specialized(Float, Double) T: ClassTag](
    other: DenseTensor[T])(implicit ev: TensorNumeric[T]): DenseTensor[T] = {
    val self = new DenseTensor[T]()
    DenseTensor.rawSet[T](self, other._storage, other._storageOffset,
      other.nDimension, other._size, other._stride)
  }

  private[tensor] def rawSet[@specialized(Float, Double) T: ClassTag](
    self: DenseTensor[T], storage: Storage[T], storageOffset: Int,
    nDimension: Int, _size: Array[Int], _stride: Array[Int]): DenseTensor[T] = {
    self._storage = storage
    require(storageOffset >= 0, "Tensor: invalid storage offset")
    self._storageOffset = storageOffset
    rawResize[T](self, nDimension, _size, _stride)
  }

  private[tensor] def rawResize[@specialized(Float, Double) T: ClassTag](
    self: DenseTensor[T], nDim: Int, _size: Array[Int], _stride: Array[Int])
  : DenseTensor[T] = {

    var hasCorrectSize = true
    var nDim_ = 0
    var d = 0
    var break = false
    while (d < nDim && !break) {
      if (_size(d) > 0) {
        nDim_ = nDim_ + 1
        if (self.nDimension > d && _size(d) != self._size(d)) {
          hasCorrectSize = false
        }
        if (self.nDimension > d && _stride != null && _stride(d) >= 0 &&
          _stride(d) != self._stride(d)) {
          hasCorrectSize = false
        }
      } else {
        break = true
      }
      d += 1
    }

    if (nDim_ != self.nDimension) hasCorrectSize = false

    if (hasCorrectSize) return self

    if (nDim_ > 0) {
      if (nDim_ != self.nDimension) {
        self._size = new Array[Int](nDim)
        self._stride = new Array[Int](nDim)
        self.nDimension = nDim
      }

      var totalSize = 1
      var d = self.nDimension - 1
      while (d >= 0) {
        self._size(d) = _size(d)
        if (_stride != null && _stride(d) >= 0) {
          self._stride(d) = _stride(d)
        } else {
          if (d == self.nDimension - 1) {
            self._stride(d) = 1
          } else {
            self._stride(d) = self._size(d + 1) * self._stride(d + 1)
          }
        }
        totalSize = totalSize + (self._size(d) - 1) * self._stride(d)

        d -= 1
      }
      if (totalSize + self._storageOffset > 0) {
        if (self._storage == null ) {
          self._storage = new ArrayStorage(new Array[T](totalSize + self._storageOffset))
        } else if (totalSize + self._storageOffset > self._storage.length) {
          self._storage.resize(totalSize + self._storageOffset)
        }
      }
    } else {
      self.nDimension = 0
    }

    self
  }

  private[tensor] def newClone[@specialized(Float, Double) T: ClassTag](self: DenseTensor[T])(
    implicit ev: TensorNumeric[T]): DenseTensor[T] = {
    val tensor = new DenseTensor[T]()
    resizeAs(tensor, self)
    copy(tensor, self)
    tensor
  }

  private[tensor] def newContiguous[@specialized(Float, Double) T: ClassTag](self: DenseTensor[T])(
    implicit ev: TensorNumeric[T]): DenseTensor[T] = {
    if (!isContiguous(self)) {
      newClone(self)
    } else {
      self
    }
  }

  private[tensor] def newSelect[@specialized(Float, Double) T: ClassTag](
    self: DenseTensor[T], _dimension: Int,
    _sliceIndex: Int)(implicit ev: TensorNumeric[T]): DenseTensor[T] = {
    val tensor = DenseTensor.newWithTensor(self)
    select(tensor, null, _dimension, _sliceIndex)
    tensor
  }

  private[tensor] def newNarrow[@specialized(Float, Double) T: ClassTag](
    self: DenseTensor[T], _dimension: Int,
    _firstIndex: Int, _size: Int)(implicit ev: TensorNumeric[T]): DenseTensor[T] = {
    val tensor = DenseTensor.newWithTensor(self)
    narrow(tensor, null, _dimension, _firstIndex, _size)
    tensor
  }

  private[tensor] def newTranspose[@specialized(Float, Double) T: ClassTag](
    self: DenseTensor[T], _dimension1: Int,
    _dimension2: Int)(implicit ev: TensorNumeric[T]): DenseTensor[T] = {
    val tensor = DenseTensor.newWithTensor(self)
    transpose(tensor, null, _dimension1, _dimension2)
    tensor
  }

  private[tensor] def resizeAs[@specialized(Float, Double) T: ClassTag](
    self: DenseTensor[T], src: Tensor[_]): Unit = {
    if (!isSameSizeAs(self, src)) rawResize(self, src.nDimension(), src.size(), null)
  }

  private[tensor] def resize[@specialized(Float, Double) T: ClassTag](
    self: DenseTensor[T], sizes: Array[Int], strides: Array[Int] = null) = {
    require(sizes != null, "invalid size")
    if (strides != null) {
      require(sizes.length == strides.length, "invalid stride")
    }
    rawResize(self, sizes.length, sizes, strides)
  }


  private[tensor] def isSameSizeAs[@specialized(Float, Double) T](
    self: DenseTensor[T], src: Tensor[_]): Boolean = {
    if (self.nDimension != src.nDimension()) {
      return false
    }

    var d = 0
    while (d < self.nDimension) {
      if (self.size(d + 1) != src.size(d + 1)) {
        return false
      }
      d += 1
    }
    return true
  }

  private[tensor] def isContiguous[@specialized(Float, Double) T](
    self: DenseTensor[T]): Boolean = {
    var s = 1
    var d = self.nDimension - 1
    while (d >= 0) {
      if (self._size(d) != 1) {
        if (s != self._stride(d)) {
          return false
        } else {
          s = s * self._size(d)
        }
      }
      d -= 1
    }
    return true
  }

  private[tensor] def size2Stride(sizes: Array[Int]): Array[Int] = {
    val strides = new Array[Int](sizes.length)
    var jump = 1
    var i = strides.length - 1
    while (i >= 0) {
      strides(i) = jump
      jump = jump * sizes(i)
      i -= 1
    }
    strides
  }

  private[tensor] def set[@specialized(Float, Double) T: ClassTag](
    self: DenseTensor[T], other: Tensor[T]): Tensor[T] = {
    if (self != other) {
      DenseTensor.rawSet(self, other.storage, other.storageOffset,
        other.nDimension, other.size, other.stride)
    } else {
      self
    }
  }

  private[tensor] def offsetFromIndice[@specialized(Float, Double) T](
    self: DenseTensor[T], indexes: Array[Int]): Int = {
    var offset = self._storageOffset
    var d = 0
    while (d < indexes.length) {
      offset = offset + (indexes(d) - 1) * self._stride(d)
      d += 1
    }
    offset
  }

  private[tensor] def select[@specialized(Float, Double) T: ClassTag](
    self: DenseTensor[T], source: Tensor[T], _dimension: Int, _sliceIndex: Int): Unit = {
    var src = source
    if (src == null) src = self
    require(src.nDimension() > 1, "cannot select on a vector")
    require(_dimension >= 0 && _dimension < src.nDimension(), "out of range")
    require(_sliceIndex >= 0 && _sliceIndex < src.size(_dimension + 1),
      s"${_sliceIndex} out of range 0 to ${src.size(_dimension + 1)}")

    set(self, src)
    narrow(self, null, _dimension, _sliceIndex, 1)

    var d = _dimension
    while (d < self.nDimension - 1) {
      self._size(d) = self._size(d + 1)
      self._stride(d) = self._stride(d + 1)
      d += 1
    }

    self.nDimension = self.nDimension - 1
  }

  private[tensor] def narrow[@specialized(Float, Double) T: ClassTag](
    self: DenseTensor[T], source: Tensor[T], _dimension: Int, _firstIndex: Int, size: Int)
  : Unit = {
    var src = source
    if (src == null) {
      src = self
    }

    require(_dimension >= 0 && _dimension < src.nDimension(), "dimension out of range")
    require(_firstIndex >= 0 && _firstIndex < src.size(_dimension + 1), "firstIndex out of range")
    require(size > 0 && _firstIndex + size <= src.size(_dimension + 1), "size out of range")

    set(self, src)

    if (_firstIndex > 0) {
      self._storageOffset = self._storageOffset + _firstIndex * self._stride(_dimension)
    }
    self._size(_dimension) = size
  }

  private[tensor] def transpose[@specialized(Float, Double) T: ClassTag](
    self: DenseTensor[T], source: Tensor[T], _dimension1: Int, _dimension2: Int): Unit = {
    var src = source
    if (src == null) src = self
    require(_dimension1 >= 0 && _dimension1 < src.nDimension, "out of range")
    require(_dimension2 >= 0 && _dimension2 < src.nDimension, "out of range")

    set(self, src)
    if (_dimension1 == _dimension2) {
      return
    }
    var z = self._stride(_dimension1)
    self._stride(_dimension1) = self._stride(_dimension2)
    self._stride(_dimension2) = z
    z = self._size(_dimension1)
    self._size(_dimension1) = self._size(_dimension2)
    self._size(_dimension2) = z
  }

  private[tensor] def get1d[@specialized(Float, Double) T](self: DenseTensor[T], x0: Int): T = {
    require(self.nDimension != 0, "tensor must have one dimension")
    require(x0 >= 0 && x0 < self._size(0), "out of range")
    self._storage(self._storageOffset + x0 * self._stride(0))
  }

  private[tensor] def get1dTensor[@specialized(Float, Double) T: ClassTag](
    self: DenseTensor[T], x0: Int)(implicit ev: TensorNumeric[T]): DenseTensor[T] = {
    new DenseTensor(new ArrayStorage(Array(get1d(self, x0))))
  }

  private[tensor] def copy[@specialized(Float, Double) T](
    self: DenseTensor[T], src: Tensor[T]): Unit = {
    require(self.nElement() == src.nElement())
    if (self.nDimension == 0) {
      return
    }
    if (self.isContiguous() && src.isContiguous() && sameStride(self.stride(), src.stride())) {
      System.arraycopy(src.storage().array(), src.storageOffset - 1, self.storage().array(),
        self.storageOffset - 1, self.nElement())
      return
    }
    val func2 = new TensorFunc4[T] {
      override def apply(data1: Array[T], offset1: Int, data2: Array[T], offset2: Int): Unit = {
        data1(offset1) = data2(offset2)
      }
    }
    DenseTensorApply.apply2[T](self, src, func2)
  }

  private[tensor] def randperm[@specialized(Float, Double) T: ClassTag](size: Int)(
    implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(size >= 1, "invalid size")

    // create an ordinal array
    val array = new Array[T](size)
    var i = 1
    while (i <= size) {
      array(i - 1) = ev.fromType[Int](i)
      i = i + 1
    }

    // Randomly exchange the elements
    i = size - 1
    while (i > 0) {
      val rand = Math.floor(RNG.uniform(0, size)).toInt
      val tmp = array(i)
      array(i) = array(rand)
      array(rand) = tmp
      i = i - 1
    }

    Tensor(new ArrayStorage(array))
  }
  private[tensor] def sameStride(l: Array[Int], r: Array[Int]): Boolean = {
    if (l.length != r.length) return false
    var i = 0
    while (i < l.length) {
      if (l(i) != r(i)) {
        return false
      }
      i += 1
    }
    return true
  }

  private[tensor] def range[@specialized(Float, Double) T: ClassTag]
  (xmin: Double, xmax: Double, step: Int = 1)(
    implicit ev: TensorNumeric[T]): Tensor[T] = {
    val newTensor = Tensor[T]()
    newTensor.range(xmin, xmax, step)
  }

  private[tensor] def ones[@specialized(Float, Double) T: ClassTag](sizes: Array[Int])(
    implicit ev: TensorNumeric[T]): Tensor[T] = {
    val length = sizes.product
    Tensor(Storage(new Array[T](length)), 1, sizes).fill(ev.fromType[Int](1))
  }

  private[tensor] def gaussian1D[@specialized(Float, Double) T: ClassTag](
      size: Int = 3,
      sigma: Double = 0.25,
      amplitude: Int = 1,
      normalize: Boolean = false,
      mean: Double = 0.5,
      tensor: Tensor[T] = null)(implicit ev: TensorNumeric[T]): Tensor[T] = {
    val gauss = if (null != tensor) {
      require(tensor.dim() == 1, "expecting 1D tensor")
      require(tensor.nElement() > 0, "expecting non-empty tensor")
      tensor
    } else {
      Tensor[T](size)
    }
    val center = mean * gauss.nElement() + 0.5

    // generate kernel
    var i = 1
    while (i <= gauss.nElement()) {
      gauss.setValue(i, ev.fromType[Double](amplitude * math.exp(-(math.pow((i - center)
        / (sigma * size), 2) / 2)))
      )
      i += 1
    }
    if (normalize) {
      gauss.div(gauss.sum())
    }
    gauss
  }
}
