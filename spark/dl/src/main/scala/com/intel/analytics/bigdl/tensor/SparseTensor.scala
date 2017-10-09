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

import breeze.linalg.{DenseMatrix, DenseVector}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table
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
 * @param _indices non-zero elements' indices
 * @param _values values of the non-zero elements
 * @param _storageOffset storageOffset
 * @param _nElement number of non-zero elements
 * @param _shape dense shape
 * @param _indicesOffset indices' offset, Default is zeros, will vary in narrowed/selected tensor.
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
     var nDimension: Int
    )(implicit ev: TensorNumeric[T]) extends Tensor[T] {

  // todo: add transpose, indices order, count from 0
  // var indices_order = Array.range(0, _shape.length)

  require(_shape.length == _indices.length, s"indices' size doesn't match tensor shape, " +
    s"indices' length is ${_indices.length} and tensor shape is ${_shape.mkString(" x ")}")

  require(_values.length == _indices(0).length, s"${_values.length()} non-zero elements should " +
    s"have indices for all elements. But indices's length is only ${_indices(0).length}")

  nDimension = _shape.length

  override def dim(): Int = nDimension

  override def setValue(d1: Int, value: T): SparseTensor.this.type = {
    throw new UnsupportedOperationException(s"Unimplemented")
    this
  }

  override def setValue(d1: Int, d2: Int, value: T): SparseTensor.this.type = {
    throw new UnsupportedOperationException(s"Unimplemented")
    this
  }

  override def setValue(d1: Int, d2: Int, d3: Int, value: T): SparseTensor.this.type = {
    throw new UnsupportedOperationException(s"Unimplemented")
    this
  }

  override def setValue(d1: Int, d2: Int, d3: Int, d4: Int, value: T): SparseTensor.this.type = {
    throw new UnsupportedOperationException(s"Unimplemented")
    this
  }

  override def setValue(d1: Int, d2: Int,
                        d3: Int, d4: Int, d5: Int, value: T): SparseTensor.this.type = {
    throw new UnsupportedOperationException(s"Unimplemented")
    this
  }

  override def unfold(dim: Int, size: Int, step: Int): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
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
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def stride(dim: Int): Int = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def fill(v: T): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def zero(): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def randn(): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def randn(mean: Double, stdv: Double): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def rand(): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def rand(lowerBound: Double, upperBound: Double): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def bernoulli(p: Double): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def transpose(dim1: Int, dim2: Int): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def t(): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def apply(index: Int): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def apply(indexes: Array[Int]): T = {
    require(indexes.length == dim())
    var index = 0
    var i = 0
    while (i < dim()) {
      index = _indices(i).array().indexOf(indexes(i) - 1, index)
      i += 1
    }
    storage().array()(index)
  }

  override def valueAt(d1: Int): T = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def valueAt(d1: Int, d2: Int): T = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def valueAt(d1: Int, d2: Int, d3: Int): T = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def valueAt(d1: Int, d2: Int, d3: Int, d4: Int): T = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def valueAt(d1: Int, d2: Int, d3: Int, d4: Int, d5: Int): T = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def apply(t: Table): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def update(index: Int, value: T): Unit = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def update(index: Int, src: Tensor[T]): Unit = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def update(indexes: Array[Int], value: T): Unit = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def update(t: Table, value: T): Unit = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def update(t: Table, src: Tensor[T]): Unit = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def update(filter: (T) => Boolean, value: T): Unit = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def isContiguous(): Boolean = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def contiguous(): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def isSameSizeAs(other: Tensor[_]): Boolean = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def resizeAs(src: Tensor[_]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def resize(sizes: Array[Int], strides: Array[Int]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def resize(size1: Int): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def resize(size1: Int, size2: Int): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def resize(size1: Int, size2: Int, size3: Int): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def resize(size1: Int, size2: Int, size3: Int, size4: Int): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def resize(size1: Int, size2: Int, size3: Int, size4: Int, size5: Int): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def select(dim: Int, index: Int): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def storage(): Storage[T] = {
    _values
  }

  override def storageOffset(): Int = {
    _storageOffset + 1
  }

  override def set(other: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def set(storage: Storage[T], storageOffset: Int,
                   sizes: Array[Int], strides: Array[Int]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def set(): Tensor[T] = {
    if (this._indices != null) {
      for (ind <- this._indices)
          ind.resize(0)
    }
    if (this._values != null) {
      this._values.resize(0)
    }
    this._nElement = 0
    this._storageOffset = 0
    this._shape = Array()
    this
  }

  override def narrow(dim: Int, index: Int, size: Int): Tensor[T] = {
    require(dim == 1, "SparseTensor.narrow only support narrow at first dimension")
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

  override def copy(other: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def apply1(func: (T) => T): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def map(other: Tensor[T], func: (T, T) => T): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def squeeze(): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def squeeze(dim: Int): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def squeezeNewTensor(): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def view(sizes: Array[Int]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def repeatTensor(sizes: Array[Int]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def expandAs(template: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def expand(sizes: Array[Int]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def split(size: Int, dim: Int): Array[Tensor[T]] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def split(dim: Int): Array[Tensor[T]] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def toBreezeVector(): DenseVector[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def toMLlibVector(): Vector = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def toBreezeMatrix(): DenseMatrix[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def toMLlibMatrix(): Matrix = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def getType(): TensorDataType = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def diff(other: Tensor[T], count: Int, reverse: Boolean): Boolean = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def addSingletonDimension(t: Tensor[T], dim: Int): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def reshape(sizes: Array[Int]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def save(path: String, overWrite: Boolean): SparseTensor.this.type = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def getTensorNumeric(): TensorNumeric[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def resize(size: Array[Int], nElement: Int): Tensor[T] = {
    if (this.nElement() < nElement) {
      storage.resize(nElement)
      if (size.length == _indices.length) {
        _indices.foreach(_.resize(nElement))
      } else if (size.length < _indices.length) {
        _indices = _indices.slice(0, size.length)
        _indices.foreach(_.resize(nElement))
      } else {
        val _addIndices = new Array[Storage[Int]](size.length - _indices.length)
        for (i <- _addIndices.indices) _addIndices(i) = Storage[Int](nElement)
        _indices ++= _addIndices
        _indices.foreach(_.resize(nElement))
      }
      _storageOffset = 0
    }
    _nElement = nElement
    _shape = size
    nDimension = size.length

    this
  }


  // scalastyle:off methodName
  override def +(s: T): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def +(t: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def -(s: T): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def -(t: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def unary_-(): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def /(s: T): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def /(t: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def *(s: T): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def *(t: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }
  // scalastyle:on methodName

  override def sum(): T = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def sum(dim: Int): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def sum(x: Tensor[T], dim: Int): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def mean(): T = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def mean(dim: Int): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def max(): T = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def max(dim: Int): (Tensor[T], Tensor[T]) = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def max(values: Tensor[T], indices: Tensor[T], dim: Int): (Tensor[T], Tensor[T]) = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def min(): T = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def min(dim: Int): (Tensor[T], Tensor[T]) = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def min(values: Tensor[T], indices: Tensor[T], dim: Int): (Tensor[T], Tensor[T]) = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def scatter(dim: Int, index: Tensor[T], src: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def gather(dim: Int, index: Tensor[T], src: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def conv2(kernel: Tensor[T], vf: Char): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def xcorr2(kernel: Tensor[T], vf: Char): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def sqrt(): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def abs(): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def add(value: T, y: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def add(y: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def add(x: Tensor[T], value: T, y: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def add(value: T): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def add(x: Tensor[T], y: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def dot(y: Tensor[T]): T = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def cmax(value: T): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def dist(y: Tensor[T], norm: Int): T = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def addcmul(value: T, tensor1: Tensor[T], tensor2: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def addcmul(tensor1: Tensor[T], tensor2: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def addcdiv(value: T, tensor1: Tensor[T], tensor2: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def sub(value: T, y: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  // Puts the result of x - value * y in current tensor
  override def sub(x: Tensor[T], value: T, y: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def sub(y: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def sub(x: Tensor[T], y: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def sub(value: T): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def cmul(y: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def cmul(x: Tensor[T], y: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def cdiv(y: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def cdiv(x: Tensor[T], y: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def mul(value: T): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def div(value: T): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def mul(x: Tensor[T], value: T): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def addmm(v1: T, M: Tensor[T], v2: T, mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def addmm(M: Tensor[T], mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def addmm(mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def addmm(v2: T, mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def addmm(v1: T, v2: T, mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def mm(mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def addr(t1: Tensor[T], t2: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def addr(v1: T, t1: Tensor[T], t2: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def addr(v1: T, t1: Tensor[T], v2: T, t2: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def addr(v1: T, t1: Tensor[T], v2: T, t2: Tensor[T], t3: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def uniform(args: T*): T = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def addmv(beta: T, vec1: Tensor[T], alpha: T,
                     mat: Tensor[T], vec2: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def addmv(beta: T, alpha: T, mat: Tensor[T], vec2: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def addmv(alpha: T, mat: Tensor[T], vec2: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def mv(mat: Tensor[T], vec2: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def baddbmm(beta: T, M: Tensor[T],
                       alpha: T, batch1: Tensor[T], batch2: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def baddbmm(beta: T, alpha: T, batch1: Tensor[T], batch2: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def baddbmm(alpha: T, batch1: Tensor[T], batch2: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def bmm(batch1: Tensor[T], batch2: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def pow(y: Tensor[T], n: T): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def pow(n: T): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def topk(k: Int, dim: Int, increase: Boolean, result: Tensor[T],
                    indices: Tensor[T]): (Tensor[T], Tensor[T]) = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def log(y: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def exp(y: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def sqrt(y: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def log1p(y: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }
  override def log(): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }
  override def exp(): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def log1p(): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def abs(x: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def norm(y: Tensor[T], value: Int, dim: Int): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def gt(x: Tensor[T], y: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def lt(x: Tensor[T], y: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def le(x: Tensor[T], y: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def eq(x: Tensor[T], y: T): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def maskedFill(mask: Tensor[T], e: T): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def maskedCopy(mask: Tensor[T], y: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def maskedSelect(mask: Tensor[T], y: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def norm(value: Int): T = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def sign(): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def ge(x: Tensor[T], value: Double): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def indexAdd(dim: Int, index: Tensor[T], y: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def index(dim: Int, index: Tensor[T], y: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def cmax(y: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def cmax(x: Tensor[T], y: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def range(xmin: Double, xmax: Double, step: Int): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def toTensor[D](implicit ev: TensorNumeric[D]): Tensor[D] = {
    if (ev.getType() == ev.getType()) {
      this.asInstanceOf[Tensor[D]]
    } else {
      throw new IllegalArgumentException(s"The type ${ev.getType().getClass}" +
        s" in toTensor[${ev.getType().getClass}] is not same" +
        s"as the numeric type ${ev.getType().getClass} of the " +
        "corresponding module, please keep them same.")
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

    println(_indices.map(_.array()).deep == other._indices.map(_.array()).deep)
    println(_indices(0).array.deep == other._indices(0).array().deep)
    println(_indices(1).array.deep == other._indices(1).array().deep)
    println(_values.array().deep == other._values.array().deep)
    println(this._shape.deep == other._shape.deep)
    println(this._nElement == other._nElement)

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
          sb.append((indices(0)(i + storageOffset) + indicesOffset)
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
        throw new UnsupportedOperationException(s"Unimplemented")
    }
  }

  override def hashCode(): Int = {
    val state = Seq(_indices, _values, _storageOffset, _nElement, _shape, nDimension)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }

  override def isEmpty: Boolean = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def isScalar: Boolean = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def value(): T = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def setValue(value: T): SparseTensor.this.type = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def applyFun[A : ClassTag](t: Tensor[A], func: (A) => T): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def zipWith[A: ClassTag, B: ClassTag](
        t1: Tensor[A],
        t2: Tensor[B],
        func: (A, B) => T): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def prod(): T = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def prod(x: Tensor[T], dim: Int): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def tanh(): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def tanh(y: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def forceFill(v: Any): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def emptyInstance(): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def forceCopy(other: Tensor[_]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  override def cast[D: ClassTag](
        castTensor: Tensor[D])(implicit ev: TensorNumeric[D]): Tensor[D] = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }
}

object SparseTensor{
  def apply[T: ClassTag](
        shape : Array[Int],
        nElement: Int = 1)(
        implicit ev: TensorNumeric[T]): SparseTensor[T] = {
    new SparseTensor(shape.map(_ => Storage[Int](nElement)), Storage(nElement),
      0, nElement,
      shape, shape.map(_ => 0), shape.length)
  }

  def apply[T: ClassTag](
      indices : Array[Array[Int]],
      values : Storage[T],
      shape : Array[Int])(
      implicit ev: TensorNumeric[T]): SparseTensor[T] = {
    new SparseTensor(indices.map(Storage(_)), values,
      0, values.length(),
      shape, shape.map(_ => 0), shape.length)
  }

  def apply[T: ClassTag](
      indices : Array[Array[Int]],
      values : Storage[T],
      shape : Array[Int],
      dimension: Int)(
    implicit ev: TensorNumeric[T]): SparseTensor[T] = {
    new SparseTensor(indices.map(Storage(_)), values,
      0, values.length(),
      shape, shape.map(_ => 0), dimension)
  }

  def apply[T: ClassTag](
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
        throw new UnsupportedOperationException(s"${denseTensor.dim()}")
    }
    SparseTensor(indices, storage, shape, shape.length)
  }

}
