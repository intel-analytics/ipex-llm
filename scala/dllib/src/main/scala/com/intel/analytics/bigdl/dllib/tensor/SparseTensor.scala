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

package com.intel.analytics.bigdl.dllib.tensor

import breeze.linalg.{DenseMatrix, DenseVector}
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.{Log4Error, Table}
import org.apache.spark.mllib.linalg.{Matrix, Vector}

import scala.reflect.ClassTag

/**
 * Tensor's sparse representation.
 *
 * To describe an SparseTensor, we need indices, values, and shape:
 * Indices means non-zero elements' indices; values means the values of the non-zero elements;
 * Shape means the dense shape of this SparseTensor.
 *
 * For example, an 2D 3x4 DenseTensor:
 *  1, 0, 0, 4
 *  0, 2, 0, 0
 *  0, 0, 3, 0
 *
 *  it's sparse representation should be
 *  indices(0) = Array(0, 0, 1, 2)
 *  indices(1) = Array(0, 3, 1, 2)
 *  values     = Array(1, 4, 2, 3)
 *  shape      = Array(3, 4)
 *
 * @param _indices non-zero elements' indices, should be zero-based and ascending.
 * @param _values values of the non-zero elements
 * @param _storageOffset storageOffset, both _values and _indices's storage offset.
 * @param _nElement number of non-zero elements
 * @param _shape dense shape
 * @param _indicesOffset indices' offset, Default is zeros, will vary in narrowed/selected tensor.
 *                       The true indices should be (_indices - _indicesOffset).
 * @param nDimension dimensions.
 * @tparam T should be Double or Float
 */
// indices is zero based.
private[tensor] class SparseTensor[@specialized(Float, Double) T: ClassTag](
    private[tensor] var _indices : Array[Storage[Int]],
    private[tensor] var _values : Storage[T],
    private[tensor] var _storageOffset: Int,
    private[tensor] var _nElement: Int,
    private[tensor] var _shape : Array[Int],
    private[tensor] var _indicesOffset : Array[Int],
    var nDimension: Int)
  (implicit ev: TensorNumeric[T]) extends Tensor[T] {

  // todo: add transpose, indices order, count from 0
  // var indices_order = Array.range(0, _shape.length)

  Log4Error.unKnowExceptionError(_shape.length == _indices.length,
    s"indices' size doesn't match tensor shape, " +
    s"indices' length is ${_indices.length} and tensor shape is ${_shape.mkString(" x ")}")

  Log4Error.unKnowExceptionError(_values.length == _indices(0).length,
    s"${_values.length()} non-zero elements should " +
    s"have indices for all elements. But indices's length is only ${_indices(0).length}")

  nDimension = _shape.length

  override def dim(): Int = nDimension

  override def setValue(d1: Int, value: T): SparseTensor.this.type = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    this
  }

  override def setValue(d1: Int, d2: Int, value: T): SparseTensor.this.type = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    this
  }

  override def setValue(d1: Int, d2: Int, d3: Int, value: T): SparseTensor.this.type = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    this
  }

  override def setValue(d1: Int, d2: Int, d3: Int, d4: Int, value: T): SparseTensor.this.type = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    this
  }

  override def setValue(
      d1: Int, d2: Int,
      d3: Int, d4: Int, d5: Int, value: T): SparseTensor.this.type = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    this
  }

  override def unfold(dim: Int, size: Int, step: Int): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    this
  }

  override def nElement(): Int = _nElement

  override def size(): Array[Int] = {
    _shape.slice(0, this.nDimension)
  }

  override def size(dim: Int): Int = {
    _shape(dim - 1)
  }

  override def stride(): Array[Int] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def stride(dim: Int): Int = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    0
  }

  override def fill(v: T): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def zero(): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def randn(): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def randn(mean: Double, stdv: Double): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def rand(): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def rand(lowerBound: Double, upperBound: Double): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def bernoulli(p: Double): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def transpose(dim1: Int, dim2: Int): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def t(): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def apply(index: Int): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def apply(indexes: Array[Int]): T = {
    Log4Error.unKnowExceptionError(indexes.length == dim(),
      s"indexes.length ${indexes.length} should match dim() ${dim()}")
    var index = 0
    var i = 0
    while (i < dim()) {
      index = _indices(i).array().indexOf(indexes(i) - 1, index)
      i += 1
    }
    storage().array()(index)
  }

  override def valueAt(d1: Int): T = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    0.asInstanceOf[T]
  }

  override def valueAt(d1: Int, d2: Int): T = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    0.asInstanceOf[T]
  }

  override def valueAt(d1: Int, d2: Int, d3: Int): T = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    0.asInstanceOf[T]
  }

  override def valueAt(d1: Int, d2: Int, d3: Int, d4: Int): T = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    0.asInstanceOf[T]
  }

  override def valueAt(d1: Int, d2: Int, d3: Int, d4: Int, d5: Int): T = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    0.asInstanceOf[T]
  }

  override def apply(t: Table): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def update(index: Int, value: T): Unit = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
  }

  override def update(index: Int, src: Tensor[T]): Unit = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
  }

  override def update(indexes: Array[Int], value: T): Unit = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
  }

  override def update(t: Table, value: T): Unit = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
  }

  override def update(t: Table, src: Tensor[T]): Unit = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
  }

  override def update(filter: (T) => Boolean, value: T): Unit = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
  }

  override def isContiguous(): Boolean = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    false
  }

  override def contiguous(): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def isSameSizeAs(other: Tensor[_]): Boolean = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    false
  }

  override def resizeAs(src: Tensor[_]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def resize(sizes: Array[Int], strides: Array[Int]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def resize(size1: Int): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def resize(size1: Int, size2: Int): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def resize(size1: Int, size2: Int, size3: Int): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def resize(size1: Int, size2: Int, size3: Int, size4: Int): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def resize(size1: Int, size2: Int, size3: Int, size4: Int, size5: Int): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def select(dim: Int, index: Int): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def storage(): Storage[T] = {
    _values
  }

  override def storageOffset(): Int = {
    _storageOffset + 1
  }

  override def set(other: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def set(
      storage: Storage[T], storageOffset: Int,
      sizes: Array[Int], strides: Array[Int]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def set(): Tensor[T] = {
    if (this._indices != null) {
      _indices.foreach(ind => ind.resize(0))
      for (i <- 0 until _indicesOffset.length) {
        _indicesOffset(i) = 0
      }
    }
    if (this._values != null) {
      this._values.resize(0)
    }
    this._nElement = 0
    this._storageOffset = 0
    this.nDimension = 0
    this._shape = Array()
    this
  }

  override def narrow(dim: Int, index: Int, size: Int): Tensor[T] = {
    Log4Error.unKnowExceptionError(dim == 1,
      "SparseTensor.narrow only support narrow at first dimension")
    dim match {
      case 1 =>
        val _index = index - 1
        val dimIndices = _indices(dim - 1)
        val indicesOffset = _indicesOffset(dim - 1)

        val nums = dimIndices.count(i => i >= _index + indicesOffset
          && i < _index + size + indicesOffset)
        val newStorageOffset = dimIndices.array().indexOf(_index + indicesOffset)
        val newShape = this.size()
        newShape(dim - 1) = size
        val newIndicesOffset = _indicesOffset.slice(0, this.nDimension)
        newIndicesOffset(dim - 1) += _index

        new SparseTensor(_indices, _values, newStorageOffset, nums, newShape,
          newIndicesOffset, newShape.length)
      case _ =>
        val _index = index - 1
        val dimIndices = _indices(dim - 1)
        val values = storage().array()

        val nums = dimIndices.count (i => i >= _index && i < _index + size)
        val newShape = this.size ()
        newShape (dim - 1) = size
        val newIndices = newShape.map (_ => new Array[Int] (nums) )
        val newStorage = Storage[T] (nums)
        val newStorageArray = newStorage.array ()
        var i = 0
        var count = 0
        while (i < storage ().array ().length) {
          if (dimIndices (i) >= _index && dimIndices (i) < (_index + size) ) {
            newStorageArray (count) = values (i)
            var dims = 0
            while (dims < this.dim () ) {
              if (dims == dim - 1) {
                newIndices(dims)(count) = _indices (dims) (i) - _index
              } else {
                newIndices(dims)(count) = _indices (dims) (i)
              }
              dims += 1
            }
            count += 1
          }
          i += 1
        }
        SparseTensor(newIndices, newStorage, newShape, newShape.length)
    }
  }

  private var nonZeroCounting: Array[Int] = _

  override def numNonZeroByRow(): Array[Int] = {
    if (null == nonZeroCounting || nonZeroCounting.length != size(1)) {
      nonZeroCounting = new Array[Int](size(1))
    }
    java.util.Arrays.fill(nonZeroCounting, 0)
    var i = _storageOffset
    while (i < _storageOffset + nElement()) {
      nonZeroCounting(_indices(0).array()(i) - _indicesOffset(0)) += 1
      i += 1
    }
    nonZeroCounting
  }

  override def copy(other: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def apply1(func: (T) => T): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def map(other: Tensor[T], func: (T, T) => T): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def squeeze(): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def squeeze(dim: Int): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def squeezeNewTensor(): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def view(sizes: Array[Int]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def repeatTensor(sizes: Array[Int]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def expandAs(template: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def expand(sizes: Array[Int]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def split(size: Int, dim: Int): Array[Tensor[T]] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def split(dim: Int): Array[Tensor[T]] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def toBreezeVector(): DenseVector[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def toMLlibVector(): Vector = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def toBreezeMatrix(): DenseMatrix[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def toMLlibMatrix(): Matrix = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def getType(): TensorDataType = {
    ev.getType()
  }

  override def diff(other: Tensor[T], count: Int, reverse: Boolean): Boolean = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    false
  }

  override def addSingletonDimension(t: Tensor[T], dim: Int): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def addMultiDimension(t: Tensor[T], dims: Array[Int]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def reshape(sizes: Array[Int]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def save(path: String, overWrite: Boolean): SparseTensor.this.type = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    this
  }

  override def getTensorNumeric(): TensorNumeric[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null.asInstanceOf[TensorNumeric[T]]
  }

  private def resizeIndices(nElement: Int): Unit = {
    var i = 0
    while (i < _indices.length) {
      _indices(i).resize(nElement + _storageOffset)
      i += 1
    }
  }

  /**
   * Notice: if size.length < dimension, will delete last (dimension - size.length)th _indices.
   * if size.length > dimension, will add indices to the front of _indices array.
   */
  override def resize(size: Array[Int], nElement: Int): Tensor[T] = {
    // if reset number of _indices
    // TODO: implement addSingletonDimension and squeeze to add/delete specified dimension.
    if (size.length < _indices.length) {
      // need to delete last (_indices.length - size.length) dimension
      _indices = _indices.slice(0, size.length)
      _indicesOffset = _indicesOffset.slice(0, size.length)
    } else if (size.length > _indices.length) {
      // add (size.length - _indices.length) dimension to the first dimension
      val _addIndices = new Array[Storage[Int]](size.length - _indices.length)
      for (i <- _addIndices.indices) _addIndices(i) = Storage[Int](nElement + _storageOffset)
      _indicesOffset = new Array[Int](size.length - _indicesOffset.length) ++ _indicesOffset
      _indices = _addIndices ++ _indices
    }

    // resize _indices's length
    if (_indices(0).length() - _storageOffset < nElement) {
      resizeIndices(nElement)
    }

    // resize _values's length
    if (storage.length() - _storageOffset < nElement) {
      storage.resize(nElement + _storageOffset)
    }
    _nElement = nElement
    _shape = size
    nDimension = size.length

    this
  }


  // scalastyle:off methodName
  override def +(s: T): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def +(t: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def -(s: T): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def -(t: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def unary_-(): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def /(s: T): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def /(t: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def *(s: T): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def *(t: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }
  // scalastyle:on methodName

  override def sum(): T = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    0.asInstanceOf[T]
  }

  override def sum(dim: Int): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def sum(x: Tensor[T], dim: Int): Tensor[T] = {
    Log4Error.unKnowExceptionError(x.dim == 1 && x.size(1) == size(1),
      s"x.dim ${x.dim} should be 1, x.size(1) ${x.size(1)} should match size(1) ${size(1)}")
    x.zero()
    var i = _storageOffset
    while (i < nElement() + _storageOffset) {
      val index = _indices(0).array()(i) - _indicesOffset(0)
      x.setValue(index, ev.plus(x.valueAt(index), _values.array()(i)))
      i += 1
    }
    x
  }

  override def mean(): T = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    0.asInstanceOf[T]
  }

  override def mean(dim: Int): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def max(): T = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    0.asInstanceOf[T]
  }

  override def max(dim: Int): (Tensor[T], Tensor[T]) = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def max(values: Tensor[T], indices: Tensor[T], dim: Int): (Tensor[T], Tensor[T]) = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def min(): T = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    0.asInstanceOf[T]
  }

  override def min(dim: Int): (Tensor[T], Tensor[T]) = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def min(values: Tensor[T], indices: Tensor[T], dim: Int): (Tensor[T], Tensor[T]) = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def scatter(dim: Int, index: Tensor[T], src: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def gather(dim: Int, index: Tensor[T], src: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def conv2(kernel: Tensor[T], vf: Char): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def xcorr2(kernel: Tensor[T], vf: Char): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def sqrt(): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def abs(): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def add(value: T, y: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def add(y: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def add(x: Tensor[T], value: T, y: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def add(value: T): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def add(x: Tensor[T], y: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def dot(y: Tensor[T]): T = {
    Log4Error.unKnowExceptionError(y.getTensorType == DenseType,
      s"y should be DenseType, but is ${y.getTensorType}")
    SparseTensorMath.vdot(y.asInstanceOf[DenseTensor[T]], this)
  }

  override def cmax(value: T): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def dist(y: Tensor[T], norm: Int): T = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    0.asInstanceOf[T]
  }

  override def addcmul(value: T, tensor1: Tensor[T], tensor2: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def addcmul(tensor1: Tensor[T], tensor2: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def addcdiv(value: T, tensor1: Tensor[T], tensor2: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def sub(value: T, y: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  // Puts the result of x - value * y in current tensor
  override def sub(x: Tensor[T], value: T, y: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def sub(y: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def sub(x: Tensor[T], y: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def sub(value: T): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def cmul(y: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def cmul(x: Tensor[T], y: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def cdiv(y: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def div(y: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def cdiv(x: Tensor[T], y: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def mul(value: T): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def div(value: T): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def mul(x: Tensor[T], value: T): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def addmm(v1: T, M: Tensor[T], v2: T, mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def addmm(M: Tensor[T], mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def addmm(mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def addmm(v2: T, mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def addmm(v1: T, v2: T, mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def mm(mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def addr(t1: Tensor[T], t2: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def addr(v1: T, t1: Tensor[T], t2: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def addr(v1: T, t1: Tensor[T], v2: T, t2: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def addr(v1: T, t1: Tensor[T], v2: T, t2: Tensor[T], t3: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def uniform(args: T*): T = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    0.asInstanceOf[T]
  }

  override def addmv(
      beta: T, vec1: Tensor[T], alpha: T,
      mat: Tensor[T], vec2: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def addmv(beta: T, alpha: T, mat: Tensor[T], vec2: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def addmv(alpha: T, mat: Tensor[T], vec2: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def mv(mat: Tensor[T], vec2: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def baddbmm(
      beta: T, M: Tensor[T],
      alpha: T, batch1: Tensor[T], batch2: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def baddbmm(beta: T, alpha: T, batch1: Tensor[T], batch2: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def baddbmm(alpha: T, batch1: Tensor[T], batch2: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def bmm(batch1: Tensor[T], batch2: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def pow(y: Tensor[T], n: T): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def pow(n: T): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def square(): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def topk(
      k: Int, dim: Int, increase: Boolean, result: Tensor[T],
      indices: Tensor[T], sortedResult: Boolean = true): (Tensor[T], Tensor[T]) = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def log(y: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def exp(y: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def sqrt(y: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def log1p(y: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def log(): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }
  override def exp(): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def log1p(): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def abs(x: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def norm(y: Tensor[T], value: Int, dim: Int): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def gt(x: Tensor[T], y: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def lt(x: Tensor[T], y: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def le(x: Tensor[T], y: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def eq(x: Tensor[T], y: T): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def maskedFill(mask: Tensor[T], e: T): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def maskedCopy(mask: Tensor[T], y: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def maskedSelect(mask: Tensor[T], y: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def norm(value: Int): T = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    0.asInstanceOf[T]
  }

  override def sign(): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def ge(x: Tensor[T], value: Double): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def indexAdd(dim: Int, index: Tensor[T], y: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def index(dim: Int, index: Tensor[T], y: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def cmax(y: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def cmax(x: Tensor[T], y: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def cmin(y: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def cmin(x: Tensor[T], y: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def range(xmin: Double, xmax: Double, step: Int): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def toTensor[D](implicit ev: TensorNumeric[D]): Tensor[D] = {
    if (ev.getType() == ev.getType()) {
      this.asInstanceOf[Tensor[D]]
    } else {
      Log4Error.invalidOperationError(false, s"The type ${ev.getType().getClass}" +
        s" in toTensor[${ev.getType().getClass}] is not same" +
        s"as the numeric type ${ev.getType().getClass} of the " +
        "corresponding module, please keep them same.")
      null
    }
  }

  override def equals(obj: Any): Boolean = {
    if (obj == null) {
      return false
    }
    if (!obj.isInstanceOf[SparseTensor[T]]) {
      return false
    }
    val other = obj.asInstanceOf[SparseTensor[T]]
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

    _indices.map(_.array()).deep == other._indices.map(_.array()).deep &&
      _values.array().deep == other._values.array().deep &&
      this._shape.deep == other._shape.deep &&
      this._nElement == other._nElement
  }

  override def toString(): String = {
    this.nDimension match {
      case 0 => s"[${this.getClass.getName} with no dimension]"
      case 1 =>
        val sb = new StringBuilder
        val indices = _indices
        val values = _values
        val storageOffset = _storageOffset
        val indicesOffset = _indicesOffset(0)
        for (i <- 0 until this.nElement)
          sb.append((indices(0)(i + storageOffset) - indicesOffset)
            + " : " + values(i + storageOffset)).append('\n')

        s"${sb}[${this.getClass.getName} of size ${this.size(1)}]"
      case 2 =>
        val sb = new StringBuilder
        val indices = _indices
        val values = _values
        val storageOffset = _storageOffset
        val indicesOffset0 = _indicesOffset(0)
        val indicesOffset1 = _indicesOffset(1)
        for (i <- 0 until this.nElement)
          sb.append("(" + (indices(0)(i + storageOffset) - indicesOffset0) + ", "
            + (indices(1)(i + storageOffset) + indicesOffset1) + ") : "
            + values(i + storageOffset)).append('\n')

        s"${sb}[${this.getClass.getName} of size ${this.size(1)}x${this.size(2)}]"
      case _ =>
        Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
        null
    }
  }

  override def hashCode(): Int = {
    val state = Seq(_indices, _values, _storageOffset, _nElement, _shape, nDimension)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }

  override def isEmpty: Boolean = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    false
  }

  override def isScalar: Boolean = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    false
  }

  override def value(): T = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    0.asInstanceOf[T]
  }

  override def setValue(value: T): SparseTensor.this.type = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def applyFun[A: ClassTag](
      t: Tensor[A],
      func: (A) => T): Tensor[T] = {
    val func2 = new TensorDiffTypeFunc4[A, T] {
      override def apply(
          data1: Array[A], index1: Int,
          data2: Array[T], index2: Int): Unit = {
        data2(index2) = func(data1(index1))
      }
    }
    SparseTensorApply.apply1[A, T](t, this, func2)
    this
  }

  override def zipWith[A: ClassTag, B: ClassTag](
      t1: Tensor[A],
      t2: Tensor[B],
      func: (A, B) => T): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def prod(): T = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    0.asInstanceOf[T]
  }

  override def prod(x: Tensor[T], dim: Int): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def tanh(): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def tanh(y: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def forceFill(v: Any): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def emptyInstance(): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def cast[@specialized(Long, Int, Short, Double, Float) D: ClassTag]
  (castTensor: Tensor[D])
    (implicit ev1: TensorNumeric[D]): Tensor[D] = {
    castTensor.getType() match {
      case FloatType =>
        castTensor.applyFun[T](this.asInstanceOf[SparseTensor[T]],
          x => ev.toType[Float](x).asInstanceOf[D])
      case DoubleType =>
        castTensor.applyFun[T](this.asInstanceOf[SparseTensor[T]],
          x => ev.toType[Double](x).asInstanceOf[D])
      case LongType =>
        castTensor.applyFun[T](this.asInstanceOf[SparseTensor[T]],
          x => ev.toType[Long](x).asInstanceOf[D])
      case IntType =>
        castTensor.applyFun[T](this.asInstanceOf[SparseTensor[T]],
          x => ev.toType[Int](x).asInstanceOf[D])
      case ShortType =>
        castTensor.applyFun[T](this.asInstanceOf[SparseTensor[T]],
          x => ev.toType[Short](x).asInstanceOf[D])
      case _ =>
        Log4Error.invalidOperationError(false, s"SparseTensor: Unsupported type")
        null
    }
    castTensor
  }

  override def getTensorType: TensorType = SparseType

  override def floor(y: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def floor(): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def ceil(): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def inv(): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def negative(x: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def reduce(dim: Int, result: Tensor[T], reducer: (T, T) => T): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def toArray(): Array[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def erf(): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def erf(t: Tensor[T]): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def erfc(): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def logGamma(): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def digamma(): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def clamp(minValue: Double, maxValue: Double): Tensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }

  override def sumSquare(): T = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    0.asInstanceOf[T]
  }

  override private[bigdl] def toQuantizedTensor: QuantizedTensor[T] = {
    Log4Error.invalidOperationError(false, s"SparseTensor: Unimplemented method")
    null
  }
}

object SparseTensor{
  private[tensor] def concat[T: ClassTag](
      dim: Int,
      tensors: Seq[Tensor[T]],
      res: Tensor[T])(implicit ev: TensorNumeric[T]): Tensor[T] = {
    Log4Error.unKnowExceptionError(dim == 1 || dim == 2,
    s"dim should be 1 or 2, but is $dim")
    var size = tensors.head.size()
    Log4Error.unKnowExceptionError(size.length <= 2,
      "Dimension larger than 2 are not supported yet!")
    tensors.foreach{tensor =>
      // todo: check size
      Log4Error.unKnowExceptionError(tensor.isInstanceOf[SparseTensor[T]],
        s"concat expect tensor is sparseTensor")
      Log4Error.unKnowExceptionError(tensor.dim() == size.length,
        s"tensor.dim() ${tensor.dim()} doesn't match size.length ${size.length}")
    }
    val dim1Concat = if (size.length == 1 && dim == 1) true else false
    if (dim1Concat) size = Array(1) ++ size
    var i = 1
    while (i < tensors.length) {
      size(dim - 1) += (if (dim1Concat) 1 else tensors(i).size(dim))
      i += 1
    }
    val totalLength = tensors.map(_.nElement()).sum

    val result = if (null == res) {
      SparseTensor(size, totalLength)
    } else {
      res.resize(size, totalLength).asInstanceOf[SparseTensor[T]]
    }
    if (dim1Concat) {
      concat(tensors.map(_.asInstanceOf[SparseTensor[T]]), result)
    }
    else {
      concat(dim, tensors.map(_.asInstanceOf[SparseTensor[T]]), result)
    }
  }

  /**
   * Concatenate a sequence of SparseTensor of 1-dim to 2-dim SparseTensor.
   *
   * @param tensors a sequence of tensors
   * @param res the resulted 2-dim SparseTensor
   * @return res
   */
  private def concat[T: ClassTag](
      tensors: Seq[SparseTensor[T]],
      res: SparseTensor[T])(implicit ev: TensorNumeric[T]): Tensor[T] = {
    val numOfIndices = res.dim()  // usually is 2
    Log4Error.unKnowExceptionError(tensors.head.dim() == 1, "Not suitable for this interface.")
    var i, offset, dimOffset = 0
    while (i < tensors.length) {
      val currentTensor = tensors(i)
      val curLength = currentTensor.nElement()
      val curTensorOffset = currentTensor.storageOffset() - 1
      // copy to concat _values
      System.arraycopy(currentTensor.storage().array(), curTensorOffset,
        res.storage().array(), offset, curLength)
      // make new Indices
      var indicesIndex = 0
      while (indicesIndex < numOfIndices) {
        if (indicesIndex == 0) {
          val storage = Storage[Int](curLength)
          val storageArray = storage.array()
          for (j <- 0 until curLength) storageArray(j) = dimOffset
          System.arraycopy(storageArray, 0, res._indices(indicesIndex).array(),
            offset, curLength)
        }
        else {
          // copy directly
          System.arraycopy(currentTensor._indices(indicesIndex - 1).array(),
            curTensorOffset, res._indices(indicesIndex).array(),
            offset, curLength)
        }
        indicesIndex += 1
      }
      offset += curLength
      dimOffset += 1
      i += 1
    }
    res
  }

  /**
   * Find index of last occurrence of value in the array from start index to end index.
   * The array should be ordered ascending.
   *
   * @param array the array will be searched.
   * @param value the element value to search for.
   * @param start start index
   * @param end last index
   * @return index of last occurrence of value
   */
  private def lastIndexOf[T: ClassTag](
      array: Array[T],
      value: T,
      start: Int,
      end: Int)(implicit ev: TensorNumeric[T]): Int = {
    if (start > end) return -1
    Log4Error.unKnowExceptionError(end <= array.length - 1,
      s"indexOf end should't exceed array size ${array.length - 1}" +
      s", but got $end")
    var i = start
    while (i < end && array(i) == value) {
      i += 1
    }
    if (array(i) == value) {
      i
    } else {
      i - 1
    }
  }

  /**
   * Find index of first occurrence of value in the array from start index to end index.
   *
   * @param array the array will be searched.
   * @param value the element value to search for.
   * @param start start index
   * @param end last index
   * @return index of first occurrence of value
   */
  private def firstIndexOf[T: ClassTag](
      array: Array[T],
      value: T,
      start: Int,
      end: Int)(implicit ev: TensorNumeric[T]): Int = {
    if (start > end) return -1
    Log4Error.unKnowExceptionError(end <= array.length - 1,
      s"indexOf end should't exceed array size ${array.length - 1}" +
      s", but got $end")
    var i = start
    while (i <= end && array(i) != value) {
      i += 1
    }
    if (i > end) {
      -1
    } else {
      i
    }
  }

  /**
   * Concatenate a sequence of SparseTensor of n-dim to n-dim SparseTensor.
   * The size at n-dim will be concated.
   *
   * @param tensors a sequence of tensors
   * @param res the resulted 2-dim SparseTensor
   * @return res
   */
  private def concat[T: ClassTag](
      dim: Int,
      tensors: Seq[SparseTensor[T]],
      res: SparseTensor[T])(implicit ev: TensorNumeric[T]): Tensor[T] = {
    val numOfIndices = res.dim()
    dim match {
      case 1 =>
        var i = 0
        var offset = 0
        var dimOffset = 0
        while (i < tensors.length) {
          val currentTensor = tensors(i)
          val curLength = currentTensor.nElement()
          val curTensorOffset = currentTensor.storageOffset() - 1

          ev.arraycopy(currentTensor.storage().array(), currentTensor.storageOffset() - 1,
            res.storage().array(), offset, currentTensor.nElement())

          var indicesIndex = 0
          while (indicesIndex < numOfIndices) {
            val indicesIndexArray = currentTensor._indices(indicesIndex).array()
            val resultIndicesArray = res._indices(indicesIndex).array()
            if (i == 0 || indicesIndex != dim - 1) {
              // copy directly
              System.arraycopy(currentTensor._indices(indicesIndex).array(),
                curTensorOffset, res._indices(indicesIndex).array(),
                offset, curLength)
            } else {
              // add size
              var j = 0
              while (j < curLength) {
                resultIndicesArray(offset + j) = indicesIndexArray(curTensorOffset + j) +
                  dimOffset
                j += 1
              }
            }
            indicesIndex += 1
          }

          offset += curLength
          dimOffset += currentTensor.size(dim)
          i += 1
        }
      case 2 =>
        var start = res._storageOffset
        var end = res._storageOffset
        val tensorsOffset = tensors.map(_.storageOffset() - 1).toArray
        val tensorsMaxIndex = tensors.map(v => v.storageOffset() + v.nElement() - 2).toArray
        var j = 0
        while (j < res.size(dim - 1)) {
          var index = 0
          var offset = 0
          while (index < tensors.size) {
            val currentTensor = tensors(index)
            val currentIndicesOffset = currentTensor._indicesOffset
            val findIndexStart =
              firstIndexOf(currentTensor._indices(0).array(), j + currentIndicesOffset(0),
                tensorsOffset(index), tensorsMaxIndex(index))
            val findIndexEnd =
              lastIndexOf(currentTensor._indices(0).array(), j + currentIndicesOffset(0),
                tensorsOffset(index), tensorsMaxIndex(index))
            val curLength = if (findIndexStart != -1 && findIndexEnd != -1) {
              findIndexEnd - findIndexStart + 1
            } else {
              0
            }

            if (0 != curLength) {
              end += curLength

              // copy values
              ev.arraycopy(currentTensor.storage().array(), tensorsOffset(index),
                res.storage().array(), start, curLength)

              // copy indices
              var indicesIndex = 0
              while (indicesIndex < numOfIndices) {
                val indicesIndexArray = currentTensor._indices(indicesIndex).array()
                val resultIndicesArray = res._indices(indicesIndex).array()
                if (indicesIndex == 0) {
                  // fill the first indices
                  res._indices(indicesIndex).fill(j, start + 1, curLength)
                } else if (index == 0) {
                  // copy directly for the first tensor's indices
                  System.arraycopy(currentTensor._indices(indicesIndex).array(),
                    tensorsOffset(index), res._indices(indicesIndex).array(), start, curLength)
                } else {
                  // add size
                  var i = 0
                  while (i < curLength) {
                    resultIndicesArray(start + i) = indicesIndexArray(tensorsOffset(index) + i) +
                      offset - currentIndicesOffset(indicesIndex)
                    i += 1
                  }
                }
                indicesIndex += 1
              }
              tensorsOffset(index) += curLength
              start = end
            }
            offset += currentTensor.size(dim)
            index += 1
          }
          j += 1
        }
    }
    res
  }

  private[tensor] def apply[T: ClassTag](
      shape : Array[Int],
      nElement: Int = 1)(
      implicit ev: TensorNumeric[T]): SparseTensor[T] = {
    new SparseTensor(shape.map(_ => Storage[Int](nElement)), Storage(nElement),
      0, nElement,
      shape, shape.map(_ => 0), shape.length)
  }

  private[tensor] def apply[T: ClassTag](
      indices : Array[Array[Int]],
      values : Storage[T],
      shape : Array[Int])(
      implicit ev: TensorNumeric[T]): SparseTensor[T] = {
    new SparseTensor(indices.map(Storage(_)), values,
      0, values.length(),
      shape, shape.map(_ => 0), shape.length)
  }

  private[tensor] def apply[T: ClassTag](
      indices : Array[Array[Int]],
      values : Storage[T],
      shape : Array[Int],
      dimension: Int)(
      implicit ev: TensorNumeric[T]): SparseTensor[T] = {
    new SparseTensor(indices.map(Storage(_)), values,
      0, values.length(),
      shape, shape.map(_ => 0), dimension)
  }

  private[tensor] def apply[T: ClassTag](
      denseTensor: Tensor[T])(implicit ev: TensorNumeric[T]): SparseTensor[T] = {
    var nonZeroElement = 0
    denseTensor.apply1{v =>
      if (v != ev.zero) nonZeroElement += 1
      v
    }
    val shape = denseTensor.size()
    val indices = shape.map(_ => new Array[Int](nonZeroElement))
    val storage = Storage[T](nonZeroElement)
    val storageArray = storage.array()
    denseTensor.dim() match {
      case 1 =>
        var sparseIndex = 0
        var i = 1
        while (i <= denseTensor.nElement()) {
          if (denseTensor.valueAt(i) != 0) {
            indices(0)(sparseIndex) = i - 1
            storageArray(sparseIndex) = denseTensor.valueAt(i)
            sparseIndex += 1
          }
          i += 1
        }
      case 2 =>
        var sparseIndex = 0
        var i = 1
        while (i <= denseTensor.size(1)) {
          var j = 1
          while (j <= denseTensor.size(2)) {
            if (denseTensor.valueAt(i, j) != 0) {
              indices(0)(sparseIndex) = i - 1
              indices(1)(sparseIndex) = j - 1
              storageArray(sparseIndex) = denseTensor.valueAt(i, j)
              sparseIndex += 1
            }
            j += 1
          }
          i += 1
        }
      case _ =>
        Log4Error.invalidOperationError(false, s"${denseTensor.dim()}")
    }
    SparseTensor(indices, storage, shape, shape.length)
  }

}
