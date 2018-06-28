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

import com.intel.analytics.bigdl.bigquant.BigQuant
import com.intel.analytics.bigdl.nn.quantized._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import java.io.{IOException, ObjectInputStream}
import scala.reflect.ClassTag

@SerialVersionUID(- 1766499387282335147L)
private[bigdl] class QuantizedTensor[T: ClassTag](
  private var _size: Array[Int],
  private var _stride: Array[Int],
  var nDimension: Int)(implicit ev: TensorNumeric[T]) extends QuantizedTensorUnsupported[T] {
  @transient private var desc = 0L
  private var internalStorage: Array[Byte] = null

  var maxOfRow: Array[T] = null
  var minOfRow: Array[T] = null
  var sumOfRow: Array[T] = null

  var params: DescParams = _

  def getStorage: Array[Byte] = {
    internalStorage
  }

  def getNativeStorage: Long = {
    desc
  }

  def release(): this.type = {
    if (desc != 0 && StorageManager.checkAndSet(desc)) {
      BigQuant.FreeMemory(desc)
    }
    desc = 0L
    this
  }

  override def equals(obj: Any): Boolean = {
    if (obj == null) {
      return false
    }

    if (!obj.isInstanceOf[QuantizedTensor[T]]) {
      return false
    }

    val other = obj.asInstanceOf[QuantizedTensor[T]]
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
    for (i <- internalStorage.indices) {
      result = internalStorage(i) == other.getStorage(i)
    }

    result
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

    if (internalStorage != null) {
      var i = 0
      while (i < internalStorage.length) {
        hash = hash * seed + internalStorage(i).toFloat.hashCode()
        i += 1
      }
    }

    hash
  }

  def this(size: Array[Int], params: DescParams)(
    implicit ev: TensorNumeric[T]) = {
    this(size, DenseTensor.size2Stride(size), size.length)
    this.params = params
    this.desc = Desc.get(params, null, 0, null, null)
  }

  def this(src: Tensor[T], descParams: DescParams)(
    implicit ev: TensorNumeric[T]) = {
    this(src.size(), src.stride(), src.nDimension())
    this.internalStorage = createInternalStorage(src)
    this.params = descParams
    this.desc = Desc.get(descParams, this.internalStorage, 0, this.maxOfRow, this.minOfRow)
  }

  def this(src: Array[Byte], size: Array[Int], max: Array[T], min: Array[T], sum: Array[T],
    descParams: DescParams)(implicit ev: TensorNumeric[T]) = {
    this(size, DenseTensor.size2Stride(size), size.length)
    require(src.length == size.product, s"size mismatch, byte array size should equal to shape")

    this.internalStorage = src
    this.maxOfRow = max
    this.minOfRow = min
    this.sumOfRow = sum
    this.params = descParams
    this.desc = Desc.get(descParams, this.internalStorage, 0, this.maxOfRow, this.minOfRow)
  }

  private def createInternalStorage(tensor: Tensor[T]): Array[Byte] = {
    val size = tensor.size(1)
    maxOfRow = new Array[T](size)
    minOfRow = new Array[T](size)
    sumOfRow = new Array[T](size)

    for (i <- 1 to size) {
      val tmp = tensor.select(1, i)
      minOfRow(i - 1) = tmp.min()
      maxOfRow(i - 1) = tmp.max()
      sumOfRow(i - 1) = tmp.sum()
    }

    val bytes = new Array[Byte](this.nElement())
    val bytesOffset = 0
    ev.getType() match {
      case FloatType =>
        Quantization.quantize(tensor.asInstanceOf[Tensor[Float]], bytes, bytesOffset)
      case _ =>
        throw new UnsupportedOperationException(s"Only support Float for quantized model")
    }

    bytes
  }

  override def getTensorType: TensorType = QuantizedType

  override def dim(): Int = nDimension

  override def size(): Array[Int] = _size

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

  override def set(): Tensor[T] = {
    internalStorage = null
    maxOfRow = null
    minOfRow = null
    sumOfRow = null
    desc = 0L
    this
  }

  /**
   * set from other tensor, it will share the storage and desc with other
   *
   * @param other the given tensor
   * @return current tensor
   */
  override def set(other: Tensor[T]): Tensor[T] = {
    if (other.isInstanceOf[QuantizedTensor[T]]) {
      val o = other.asInstanceOf[QuantizedTensor[T]]

      this.internalStorage = o.getStorage
      this.params = o.params
      this.desc = o.getNativeStorage

      this.maxOfRow = o.maxOfRow
      this.minOfRow = o.minOfRow
      this.sumOfRow = o.sumOfRow

      this._size = o.size()
      this._stride = o.stride()
      this.nDimension = o.nDimension

    } else {
      throw new UnsupportedOperationException(s"can't set from other type of tensor.")
    }
    this
  }

  /**
   * copy from another QuantizedTensor, it will a new storage and new desc
   *
   * @param other source tensor
   * @return current tensor
   */
  override def copy(other: Tensor[T]): Tensor[T] = {
    if (other.isInstanceOf[QuantizedTensor[T]] && other.size().deep == this.size().deep) {
      val quantizedTensor = other.asInstanceOf[QuantizedTensor[T]]

      if (internalStorage != null) {
        internalStorage = new Array[Byte](other.nElement())
      }

      System.arraycopy(quantizedTensor.getStorage, 0, internalStorage, 0, this.nElement())

      params = quantizedTensor.params.copy()

      val length = quantizedTensor.maxOfRow.length
      maxOfRow = new Array[T](length)
      System.arraycopy(quantizedTensor.maxOfRow, 0, maxOfRow, 0, length)

      minOfRow = new Array[T](length)
      System.arraycopy(quantizedTensor.minOfRow, 0, minOfRow, 0, length)

      sumOfRow = new Array[T](length)
      System.arraycopy(quantizedTensor.sumOfRow, 0, sumOfRow, 0, length)

      this.desc = Desc.get(params, internalStorage, 0, this.maxOfRow, this.minOfRow)
    } else {
      throw new UnsupportedOperationException(s"can't set from other type of tensor.")
    }

    this
  }

  override def getTensorNumeric(): TensorNumeric[T] = ev

  override def toQuantizedTensor: QuantizedTensor[T] = this.asInstanceOf[QuantizedTensor[T]]

  @throws(classOf[IOException])
  private def readObject(in: ObjectInputStream): Unit = {
    in.defaultReadObject()

    this.desc = Desc.get(params, internalStorage, 0, maxOfRow, minOfRow)
  }
}

object QuantizedTensor {
  def apply[@specialized(Float, Double) T: ClassTag](size: Array[Int], params: DescParams)(
    implicit ev: TensorNumeric[T]): QuantizedTensor[T] =
    new QuantizedTensor[T](size, params)

  def apply[@specialized(Float, Double) T: ClassTag](src: Tensor[T], descParams: DescParams)(
    implicit ev: TensorNumeric[T]): QuantizedTensor[T] = {
    new QuantizedTensor[T](src, descParams)
  }

  def apply[@specialized(Float, Double) T: ClassTag](src: Array[Byte], max: Array[T], min: Array[T],
    sum: Array[T], size: Array[Int], descParams: DescParams)(
    implicit ev: TensorNumeric[T]): QuantizedTensor[T] = {
    new QuantizedTensor[T](src, size, max, min, sum, descParams)
  }
}

object QuantizedDummyTensor {
  def apply[@specialized(Float, Double) T: ClassTag]()(
    implicit ev: TensorNumeric[T]): QuantizedTensor[T] = {
    QuantizedTensor[T](Tensor(1, 1), LinearWeightParams(1, 1))
  }
}
