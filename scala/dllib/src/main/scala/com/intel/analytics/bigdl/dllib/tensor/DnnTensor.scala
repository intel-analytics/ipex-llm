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
import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.nn.mkldnn.{MemoryOwner, Releasable}
import com.intel.analytics.bigdl.tensor.DnnTensor.DnnTensorUnsupportOperations
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Matrix
import scala.reflect.ClassTag

class DnnTensor[T: ClassTag](
  private var _storage: DnnStorage[T],
  private var sizes: Array[Int]
) (implicit ev: TensorNumeric[T], owner: MemoryOwner)
  extends DnnTensorUnsupportOperations[T] with Releasable {

  owner.registerResource(this)
  // performance regression, the sizes.product will make the performance downgrade.
  private val _nElement: Int = sizes.product

  override def nElement(): Int = _nElement

  override def copy(other: Tensor[T]): Tensor[T] = {
    other match {
      case t: DenseTensor[_] =>
        require(DnnTensor.noTransposed(t), "dense tensor should not be transposed")
        require(this.nElement() == other.nElement(), "tensor elements number must be same")
        this._storage.copy(other.storage(), 0, other.storageOffset() - 1, this.nElement())
      case t: DnnTensor[_] =>
        require(this.nElement() == other.nElement(), "tensor elements number must be same")
        this._storage.copy(other.storage(), 0, 0, this.nElement())
      case _ => throw new UnsupportedOperationException(
        "Only support copy from dense tensor and dnn tensor")
    }
    this
  }

  def release(): Unit = {
    _storage.release()
  }

  def storageAddress(): Long = _storage.ptr.address

  def isReleased(): Boolean = _storage.isReleased()

  override def storage(): Storage[T] = _storage

  override def resize(s: Array[Int], stride: Array[Int] = null): this.type = {
    require(stride == null, "dnn tensor doesn't have stride")
    if (s.product > nElement()) {
      _storage.release()
      _storage = new DnnStorage[T](s.product)
    }
    this.sizes = s.clone()
    this
  }

  override def resize(s: Int): this.type = {
    if (s > nElement()) {
      _storage.release()
      _storage = new DnnStorage[T](s)
    }
    this.sizes = Array(s)
    this
  }

  override def add(x: Tensor[T]): Tensor[T] = {
    require(x.isInstanceOf[DnnTensor[_]], "Just support two dnn tensor add")
    Memory.SAdd(this.nElement(), this._storage.ptr.address, 0,
      x.asInstanceOf[DnnTensor[T]]._storage.ptr.address, 0, this._storage.ptr.address, 0)
    this
  }

  override def zero(): Tensor[T] = {
    Memory.Zero(this._storage.ptr.address, this.nElement(), DnnStorage.FLOAT_BYTES)
    this
  }

  def axpby(a: Float, b: Float, to: DnnTensor[T]): Unit = {
    val x = this._storage.ptr.address
    val y = to._storage.ptr.address
    Memory.Axpby(this.nElement(), a, x, b, y)
  }

  def scale(from: DnnTensor[T], scal: Float): Unit = {
    val length = this.nElement()
    Memory.Scale(length, scal, from._storage.ptr.address, this._storage.ptr.address)
  }

  override def toTensor[D](implicit ev: TensorNumeric[D]): DnnTensor[D] = {
    this.asInstanceOf[DnnTensor[D]]
  }

  override def size(): Array[Int] = sizes.clone()

  override def size(d: Int): Int = sizes(d - 1)

  override def dim(): Int = size().length

  override def nDimension(): Int = size().length

  override def getTensorType: TensorType = MklDnnType

  override def equals(obj: Any): Boolean = {
    if (obj == null) {
      return false
    }
    if (!obj.isInstanceOf[DnnTensor[T]]) {
      return false
    }
    val other = obj.asInstanceOf[DnnTensor[T]]

    if (this.size().deep != other.size().deep) {
      return false
    }

    if (this._storage.ptr != other._storage.ptr) {
      return false
    }

    true
  }

  override def getType(): TensorDataType = {
    ev.getType()
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

    hash = hash * seed + this._storage.ptr.hashCode()

    hash
  }

  override def set(): Tensor[T] = {
    // TODO we will do nothing. the behavior is not the same with DenseTensor
    this
  }

  override def set(other: Tensor[T]): Tensor[T] = {
    require(other.isInstanceOf[DnnTensor[T]], s"only support to set DnnTensor")
    this._storage.release()
    this._storage = other.storage().asInstanceOf[DnnStorage[T]]
    this
  }

  override def toString: String = {
    ev.getType() match {
      case FloatType =>
        if (size().product != this.nElement()) {
          val dense = Tensor[Float](Array(this.nElement()))
          Memory.CopyPtr2Array(this.storageAddress(), 0, dense.storage().array(),
            0, nElement(), 4)
          dense.toString
        } else {
          val dense = Tensor[Float](size())
          dense.copy(this.asInstanceOf[DnnTensor[Float]])
          dense.toString
        }
      case ByteType =>
        val array = new Array[Byte](nElement())
        Memory.CopyPtr2ByteArray(this.asInstanceOf[DnnTensor[Byte]].storageAddress(), 0,
          array, 0, nElement(), 1)
        array.mkString("\t")
      case IntType =>
        val array = new Array[Int](nElement())
        Memory.CopyPtr2IntArray(this.storageAddress(), 0, array, 0, nElement(), 4)
        array.mkString("\t")
      case _ => "unknown type"
    }
  }
}

object DnnTensor {
  // scalastyle:off
  private def ???(): Nothing = {
    throw new UnsupportedOperationException("DnnTensor doesn't support this operation")
  }
  // scalastyle:on

  private[tensor] def noTransposed(t: DenseTensor[_]): Boolean = {
    var product = 1
    var i = t.dim()
    while(i > 0) {
      if (product != t.stride(i)) return false
      product *= t.size(i)
      i -= 1
    }
    return true
  }

  def apply[T: ClassTag](sizes: Array[Int])(
    implicit ev: TensorNumeric[T], owner: MemoryOwner): DnnTensor[T] = {
    val storage = new DnnStorage[T](sizes.product)
    new DnnTensor[T](storage, sizes)
  }

  def apply[T: ClassTag](sizes: Array[Int], realSize: Long)(
    implicit ev: TensorNumeric[T], owner: MemoryOwner): DnnTensor[T] = {
    val storage = new DnnStorage[T](realSize.toInt) // FIXME if size more than int ?
    new DnnTensor[T](storage, sizes)
  }

  def apply[T: ClassTag](d1: Int)(
    implicit ev: TensorNumeric[T], owner: MemoryOwner): DnnTensor[T] = {
    val storage = new DnnStorage[T](d1)
    new DnnTensor[T](storage, Array(d1))
  }

  def apply[T: ClassTag](d1: Int, d2: Int)(
    implicit ev: TensorNumeric[T], owner: MemoryOwner): DnnTensor[T] = {
    val storage = new DnnStorage[T](d1 * d2)
    new DnnTensor[T](storage, Array(d1, d2))
  }

  def apply[T: ClassTag](d1: Int, d2: Int, d3: Int)(
    implicit ev: TensorNumeric[T], owner: MemoryOwner): DnnTensor[T] = {
    val storage = new DnnStorage[T](d1 * d2 * d3)
    new DnnTensor[T](storage, Array(d1, d2, d3))
  }

  def apply[T: ClassTag](d1: Int, d2: Int, d3: Int, d4: Int)(
    implicit ev: TensorNumeric[T], owner: MemoryOwner): DnnTensor[T] = {
    val storage = new DnnStorage[T](d1 * d2 * d3 * d4)
    new DnnTensor[T](storage, Array(d1, d2, d3, d4))
  }

  def apply[T: ClassTag](d1: Int, d2: Int, d3: Int, d4: Int, d5: Int)(
    implicit ev: TensorNumeric[T], owner: MemoryOwner): DnnTensor[T] = {
    val storage = new DnnStorage[T](d1 * d2 * d3 * d4 * d5)
    new DnnTensor[T](storage, Array(d1, d2, d3, d4, d5))
  }

  class DnnTensorUnsupportOperations[T: ClassTag](implicit ev: TensorNumeric[T]) extends Tensor[T] {
    // scalastyle:off
    override def isEmpty: Boolean = ???
    override def isScalar: Boolean = ???
    override def nDimension(): Int = ???
    override def dim(): Int = ???
    override def size(): Array[Int] = ???
    override def size(dim: Int): Int = ???
    override def stride(): Array[Int] = ???
    override def stride(dim: Int): Int = ???
    override def fill(v: T): Tensor[T] = ???
    override def forceFill(v: Any): Tensor[T] = ???
    override def zero(): Tensor[T] = ???
    override def randn(): Tensor[T] = ???
    override def randn(mean: Double, stdv: Double): Tensor[T] = ???
    override def rand(): Tensor[T] = ???
    override def rand(lowerBound: Double, upperBound: Double): Tensor[T] = ???
    override def bernoulli(p: Double): Tensor[T] = ???
    override def transpose(dim1: Int, dim2: Int): Tensor[T] = ???
    override def t(): Tensor[T] = ???
    override def apply(index: Int): Tensor[T] = ???
    override def apply(indexes: Array[Int]): T = ???
    override def value(): T = ???
    override def valueAt(d1: Int): T = ???
    override def valueAt(d1: Int, d2: Int): T = ???
    override def valueAt(d1: Int, d2: Int, d3: Int): T = ???
    override def valueAt(d1: Int, d2: Int, d3: Int, d4: Int): T = ???
    override def valueAt(d1: Int, d2: Int, d3: Int, d4: Int, d5: Int): T = ???
    override def apply(t: Table): Tensor[T] = ???
    override def update(index: Int, value: T): Unit = ???
    override def update(index: Int, src: Tensor[T]): Unit = ???
    override def update(indexes: Array[Int], value: T): Unit = ???
    override def setValue(value: T): DnnTensorUnsupportOperations.this.type = ???
    override def setValue(d1: Int, value: T): DnnTensorUnsupportOperations.this.type = ???
    override def setValue(d1: Int, d2: Int, value: T): DnnTensorUnsupportOperations.this.type = ???
    override def setValue(d1: Int, d2: Int, d3: Int, value: T): DnnTensorUnsupportOperations.this.type = ???
    override def setValue(d1: Int, d2: Int, d3: Int, d4: Int, value: T): DnnTensorUnsupportOperations.this.type = ???
    override def setValue(d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, value: T): DnnTensorUnsupportOperations.this.type = ???
    override def update(t: Table, value: T): Unit = ???
    override def update(t: Table, src: Tensor[T]): Unit = ???
    override def update(filter: (T) => Boolean, value: T): Unit = ???
    override def isContiguous(): Boolean = ???
    override def contiguous(): Tensor[T] = ???
    override def isSameSizeAs(other: Tensor[_]): Boolean = ???
    override def emptyInstance(): Tensor[T] = ???
    override def resizeAs(src: Tensor[_]): Tensor[T] = ???
    override def cast[D: ClassManifest](castTensor: Tensor[D])(implicit ev: TensorNumeric[D]): Tensor[D] = ???
    override def resize(sizes: Array[Int], strides: Array[Int]): Tensor[T] = ???
    override def resize(size1: Int): Tensor[T] = ???
    override def resize(size1: Int, size2: Int): Tensor[T] = ???
    override def resize(size1: Int, size2: Int, size3: Int): Tensor[T] = ???
    override def resize(size1: Int, size2: Int, size3: Int, size4: Int): Tensor[T] = ???
    override def resize(size1: Int, size2: Int, size3: Int, size4: Int, size5: Int): Tensor[T] = ???
    override def nElement(): Int = ???
    override def select(dim: Int, index: Int): Tensor[T] = ???
    override def storage(): Storage[T] = ???
    override def storageOffset(): Int = ???
    override def set(other: Tensor[T]): Tensor[T] = ???
    override def set(storage: Storage[T], storageOffset: Int, sizes: Array[Int], strides: Array[Int]): Tensor[T] = ???
    override def set(): Tensor[T] = ???
    override def narrow(dim: Int, index: Int, size: Int): Tensor[T] = ???
    override def copy(other: Tensor[T]): Tensor[T] = ???
    override def applyFun[A: ClassManifest](t: Tensor[A], func: (A) => T): Tensor[T] = ???
    override def apply1(func: (T) => T): Tensor[T] = ???
    override def zipWith[A: ClassManifest, B: ClassManifest](t1: Tensor[A], t2: Tensor[B], func: (A, B) => T): Tensor[T] = ???
    override def map(other: Tensor[T], func: (T, T) => T): Tensor[T] = ???
    override def squeeze(): Tensor[T] = ???
    override def squeeze(dim: Int): Tensor[T] = ???
    override def squeezeNewTensor(): Tensor[T] = ???
    override def view(sizes: Array[Int]): Tensor[T] = ???
    override def unfold(dim: Int, size: Int, step: Int): Tensor[T] = ???
    override def repeatTensor(sizes: Array[Int]): Tensor[T] = ???
    override def expandAs(template: Tensor[T]): Tensor[T] = ???
    override def expand(sizes: Array[Int]): Tensor[T] = ???
    override def split(size: Int, dim: Int): Array[Tensor[T]] = ???
    override def split(dim: Int): Array[Tensor[T]] = ???
    override def toBreezeVector(): DenseVector[T] = ???
    override def toMLlibVector(): linalg.Vector = ???
    override def toBreezeMatrix(): DenseMatrix[T] = ???
    override def toMLlibMatrix(): Matrix = ???
    override def getType(): TensorDataType = ???
    override def diff(other: Tensor[T], count: Int, reverse: Boolean): Boolean = ???
    override def addSingletonDimension(t: Tensor[T], dim: Int): Tensor[T] = ???
    override def addMultiDimension(t: Tensor[T], dims: Array[Int]): Tensor[T] = ???
    override def reshape(sizes: Array[Int]): Tensor[T] = ???
    override def save(path: String, overWrite: Boolean): DnnTensorUnsupportOperations.this.type = ???
    override def getTensorNumeric(): TensorNumeric[T] = ???
    override def getTensorType: TensorType = ???
    override def toArray(): Array[T] = ???
    override def +(s: T): Tensor[T] = ???
    override def +(t: Tensor[T]): Tensor[T] = ???
    override def -(s: T): Tensor[T] = ???
    override def -(t: Tensor[T]): Tensor[T] = ???
    override def unary_-(): Tensor[T] = ???
    override def /(s: T): Tensor[T] = ???
    override def /(t: Tensor[T]): Tensor[T] = ???
    override def *(s: T): Tensor[T] = ???
    override def *(t: Tensor[T]): Tensor[T] = ???
    override def sum(): T = ???
    override def prod(): T = ???
    override def prod(x: Tensor[T], dim: Int): Tensor[T] = ???
    override def sum(dim: Int): Tensor[T] = ???
    override def sum(x: Tensor[T], dim: Int): Tensor[T] = ???
    override def mean(): T = ???
    override def mean(dim: Int): Tensor[T] = ???
    override def max(): T = ???
    override def max(dim: Int): (Tensor[T], Tensor[T]) = ???
    override def max(values: Tensor[T], indices: Tensor[T], dim: Int): (Tensor[T], Tensor[T]) = ???
    override def min(): T = ???
    override def min(dim: Int): (Tensor[T], Tensor[T]) = ???
    override def min(values: Tensor[T], indices: Tensor[T], dim: Int): (Tensor[T], Tensor[T]) = ???
    override def scatter(dim: Int, index: Tensor[T], src: Tensor[T]): Tensor[T] = ???
    override def gather(dim: Int, index: Tensor[T], src: Tensor[T]): Tensor[T] = ???
    override def conv2(kernel: Tensor[T], vf: Char): Tensor[T] = ???
    override def xcorr2(kernel: Tensor[T], vf: Char): Tensor[T] = ???
    override def sqrt(): Tensor[T] = ???
    override def tanh(): Tensor[T] = ???
    override def abs(): Tensor[T] = ???
    override def add(value: T, y: Tensor[T]): Tensor[T] = ???
    override def add(y: Tensor[T]): Tensor[T] = ???
    override def add(x: Tensor[T], value: T, y: Tensor[T]): Tensor[T] = ???
    override def add(value: T): Tensor[T] = ???
    override def add(x: Tensor[T], y: Tensor[T]): Tensor[T] = ???
    override def dot(y: Tensor[T]): T = ???
    override def cmax(value: T): Tensor[T] = ???
    override def dist(y: Tensor[T], norm: Int): T = ???
    override def addcmul(value: T, tensor1: Tensor[T], tensor2: Tensor[T]): Tensor[T] = ???
    override def addcmul(tensor1: Tensor[T], tensor2: Tensor[T]): Tensor[T] = ???
    override def addcdiv(value: T, tensor1: Tensor[T], tensor2: Tensor[T]): Tensor[T] = ???
    override def sub(value: T, y: Tensor[T]): Tensor[T] = ???
    override def sub(x: Tensor[T], value: T, y: Tensor[T]): Tensor[T] = ???
    override def sub(y: Tensor[T]): Tensor[T] = ???
    override def sub(x: Tensor[T], y: Tensor[T]): Tensor[T] = ???
    override def sub(value: T): Tensor[T] = ???
    override def cmul(y: Tensor[T]): Tensor[T] = ???
    override def cmul(x: Tensor[T], y: Tensor[T]): Tensor[T] = ???
    override def cdiv(y: Tensor[T]): Tensor[T] = ???
    override def cdiv(x: Tensor[T], y: Tensor[T]): Tensor[T] = ???
    override def mul(value: T): Tensor[T] = ???
    override def div(value: T): Tensor[T] = ???
    override def div(y: Tensor[T]): Tensor[T] = ???
    override def mul(x: Tensor[T], value: T): Tensor[T] = ???
    override def addmm(v1: T, M: Tensor[T], v2: T, mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] = ???
    override def addmm(M: Tensor[T], mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] = ???
    override def addmm(mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] = ???
    override def addmm(v2: T, mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] = ???
    override def addmm(v1: T, v2: T, mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] = ???
    override def mm(mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] = ???
    override def addr(t1: Tensor[T], t2: Tensor[T]): Tensor[T] = ???
    override def addr(v1: T, t1: Tensor[T], t2: Tensor[T]): Tensor[T] = ???
    override def addr(v1: T, t1: Tensor[T], v2: T, t2: Tensor[T]): Tensor[T] = ???
    override def addr(v1: T, t1: Tensor[T], v2: T, t2: Tensor[T], t3: Tensor[T]): Tensor[T] = ???
    override def uniform(args: T*): T = ???
    override def addmv(beta: T, vec1: Tensor[T], alpha: T, mat: Tensor[T], vec2: Tensor[T]): Tensor[T] = ???
    override def addmv(beta: T, alpha: T, mat: Tensor[T], vec2: Tensor[T]): Tensor[T] = ???
    override def addmv(alpha: T, mat: Tensor[T], vec2: Tensor[T]): Tensor[T] = ???
    override def mv(mat: Tensor[T], vec2: Tensor[T]): Tensor[T] = ???
    override def baddbmm(beta: T, M: Tensor[T], alpha: T, batch1: Tensor[T], batch2: Tensor[T]): Tensor[T] = ???
    override def baddbmm(beta: T, alpha: T, batch1: Tensor[T], batch2: Tensor[T]): Tensor[T] = ???
    override def baddbmm(alpha: T, batch1: Tensor[T], batch2: Tensor[T]): Tensor[T] = ???
    override def bmm(batch1: Tensor[T], batch2: Tensor[T]): Tensor[T] = ???
    override def pow(y: Tensor[T], n: T): Tensor[T] = ???
    override def pow(n: T): Tensor[T] = ???
    override def square(): Tensor[T] = ???
    override def floor(y: Tensor[T]): Tensor[T] = ???
    override def floor(): Tensor[T] = ???
    override def ceil(): Tensor[T] = ???
    override def inv(): Tensor[T] = ???
    override def erf(): Tensor[T] = ???
    override def erfc(): Tensor[T] = ???
    override def logGamma(): Tensor[T] = ???
    override def digamma(): Tensor[T] = ???
    override def topk(k: Int, dim: Int, increase: Boolean, result: Tensor[T], indices: Tensor[T], sortedResult: Boolean): (Tensor[T], Tensor[T]) = ???
    override def log(y: Tensor[T]): Tensor[T] = ???
    override def exp(y: Tensor[T]): Tensor[T] = ???
    override def sqrt(y: Tensor[T]): Tensor[T] = ???
    override def tanh(y: Tensor[T]): Tensor[T] = ???
    override def log1p(y: Tensor[T]): Tensor[T] = ???
    override def log(): Tensor[T] = ???
    override def exp(): Tensor[T] = ???
    override def log1p(): Tensor[T] = ???
    override def abs(x: Tensor[T]): Tensor[T] = ???
    override def norm(y: Tensor[T], value: Int, dim: Int): Tensor[T] = ???
    override def gt(x: Tensor[T], y: Tensor[T]): Tensor[T] = ???
    override def lt(x: Tensor[T], y: Tensor[T]): Tensor[T] = ???
    override def le(x: Tensor[T], y: Tensor[T]): Tensor[T] = ???
    override def eq(x: Tensor[T], y: T): Tensor[T] = ???
    override def maskedFill(mask: Tensor[T], e: T): Tensor[T] = ???
    override def maskedCopy(mask: Tensor[T], y: Tensor[T]): Tensor[T] = ???
    override def maskedSelect(mask: Tensor[T], y: Tensor[T]): Tensor[T] = ???
    override def norm(value: Int): T = ???
    override def sign(): Tensor[T] = ???
    override def ge(x: Tensor[T], value: Double): Tensor[T] = ???
    override def indexAdd(dim: Int, index: Tensor[T], y: Tensor[T]): Tensor[T] = ???
    override def index(dim: Int, index: Tensor[T], y: Tensor[T]): Tensor[T] = ???
    override def cmax(y: Tensor[T]): Tensor[T] = ???
    override def cmin(y: Tensor[T]): Tensor[T] = ???
    override def cmax(x: Tensor[T], y: Tensor[T]): Tensor[T] = ???
    override def cmin(x: Tensor[T], y: Tensor[T]): Tensor[T] = ???
    override def range(xmin: Double, xmax: Double, step: Int): Tensor[T] = ???
    override def negative(x: Tensor[T]): Tensor[T] = ???
    override def reduce(dim: Int, result: Tensor[T], reducer: (T, T) => T): Tensor[T] = ???
    override def sumSquare(): T = ???
    override def clamp(min: Double, max: Double): Tensor[T] = ???
    override def toTensor[D](implicit ev: TensorNumeric[D]): Tensor[D] = ???
    override private[bigdl] def toQuantizedTensor = ???
    // scalastyle: on
  }
}
