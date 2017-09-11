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
import com.intel.analytics.bigdl.bigquant.BigQuant
import com.intel.analytics.bigdl.nn.quantized._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table
import java.nio.ByteBuffer
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Matrix
import scala.reflect.ClassTag

@SerialVersionUID(- 1766499387282335147L)
private[bigdl] class QuantizedTensor[@specialized(Float) T: ClassTag](
  private var _size: Array[Int],
  private var _stride: Array[Int],
  var nDimension: Int)(implicit ev: TensorNumeric[T]) extends QuantTensorUnsupported[T] {
  @transient private var desc = 0L
  private var value: Array[Byte] = null

  var maxOfRow: Array[T] = null
  var minOfRow: Array[T] = null
  var sumOfRow: Array[T] = null

  var params: DescParams = _
  var descType: DescType = _

  def getStorage: Array[Byte] = {
    value
  }

  def getNativeStorage: Long = {
    desc
  }

  def release(): this.type = {
    if (desc != 0) {
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
    for (i <- value.indices) {
      result = value(i) == other.getStorage(i)
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

    if (value != null) {
      var i = 0
      while (i < value.length) {
        hash = hash * seed + value(i).toFloat.hashCode()
        i += 1
      }
    }

    hash
  }

  def this()(implicit ev: TensorNumeric[T]) = this(null, null, 0)

  def this(d1: Int)(implicit ev: TensorNumeric[T]) = this(Array(d1), Array(1), 1)

  def this(d1: Int, d2: Int)(implicit ev: TensorNumeric[T]) = this(Array(d1, d2), Array(d2, 1), 2)

  def this(d1: Int, d2: Int, d3: Int)(implicit ev: TensorNumeric[T]) =
    this(Array(d1, d2, d3), Array(d3 * d2, d3, 1), 3)

  def this(d1: Int, d2: Int, d3: Int, d4: Int)(implicit ev: TensorNumeric[T]) =
    this(Array(d1, d2, d3, d4), Array(d4 * d3 * d2, d4 * d3, d4, 1), 4)

  def this(d1: Int, d2: Int, d3: Int, d4: Int, d5: Int)(implicit ev: TensorNumeric[T]) =
    this(Array(d1, d2, d3, d4, d5), Array(d5 * d4 * d3 * d2, d5 * d4 * d3, d5 * d4, d5, 1), 5)

  def this(size: Array[Int])(implicit ev: TensorNumeric[T]) =
    this(size, DenseTensor.size2Stride(size), size.length)

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

  // TODO rename init
  def init(params: DescParams, descType: DescType): Unit = {
    // two cases:
    // weight init
    // the input attributes have been changed
    if (this.desc == 0L || this.params != params) {
      release()
      this.desc = Desc.get(params, descType, this)
    }
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

  override def resize(sizes: Array[Int], strides: Array[Int]): Tensor[T] = {
    _size = sizes
    _stride = strides
    this
  }

  override def resize(size1: Int): Tensor[T] = {
    if (this.nDimension != 1 || this.size(1) != size1) {
      resize(Array(size1))
    } else {
      this
    }
  }

  override def resize(size1: Int, size2: Int): Tensor[T] = {
    if (this.nDimension != 2 || this.size(1) != size1 || this.size(2) != size2) {
      resize(Array(size1, size2))
    } else {
      this
    }
  }

  override def resize(size1: Int, size2: Int, size3: Int): Tensor[T] = {
    if (this.nDimension != 3 || this.size(1) != size1 || this.size(2) != size2 ||
      this.size(3) != size3) {
      resize(Array(size1, size2, size3))
    } else {
      this
    }
  }

  override def resize(size1: Int, size2: Int, size3: Int, size4: Int): Tensor[T] = {
    if (this.nDimension != 4 || this.size(1) != size1 || this.size(2) != size2 ||
      this.size(3) != size3 ||
      this.size(4) != size4) {
      resize(Array(size1, size2, size3, size4))
    } else {
      this
    }
  }

  override def resize(size1: Int, size2: Int, size3: Int, size4: Int, size5: Int): Tensor[T] = {
    if (this.nDimension != 5 || this.size(1) != size1 || this.size(2) != size2 ||
      this.size(3) != size3 || this.size(4) != size4 || this.size(5) != size5) {
      resize(Array(size1, size2, size3, size4, size5))
    } else {
      this
    }
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
    value = null
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

      this.value = o.getStorage
      this.params = o.params
      this.descType = o.descType
      this.desc = o.getNativeStorage

      this.maxOfRow = o.maxOfRow
      this.minOfRow = o.minOfRow
      this.sumOfRow = o.sumOfRow

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

      if (value != null) {
        value = new Array[Byte](other.nElement())
      }

      System.arraycopy(quantizedTensor.getStorage, 0, value, 0, this.nElement())

      params = quantizedTensor.params.copy()
      descType = quantizedTensor.descType

      val length = quantizedTensor.maxOfRow.length
      maxOfRow = new Array[T](length)
      System.arraycopy(quantizedTensor.maxOfRow, 0, maxOfRow, 0, length)

      minOfRow = new Array[T](length)
      System.arraycopy(quantizedTensor.minOfRow, 0, minOfRow, 0, length)

      sumOfRow = new Array[T](length)
      System.arraycopy(quantizedTensor.sumOfRow, 0, sumOfRow, 0, length)

      init(params, descType)
    } else {
      throw new UnsupportedOperationException(s"can't set from other type of tensor.")
    }

    this
  }
}

object QuantizedTensor {
  def apply[@specialized(Float, Double) T: ClassTag]()(
    implicit ev: TensorNumeric[T]): QuantizedTensor[T] = new QuantizedTensor[T]()

  def apply[@specialized(Float, Double) T: ClassTag](d1: Int)(
    implicit ev: TensorNumeric[T]): QuantizedTensor[T] = new QuantizedTensor[T](d1)

  def apply[@specialized(Float, Double) T: ClassTag](d1: Int, d2: Int)(
    implicit ev: TensorNumeric[T]): QuantizedTensor[T] = new QuantizedTensor[T](d1, d2)

  def apply[@specialized(Float, Double) T: ClassTag](d1: Int, d2: Int, d3: Int)(
    implicit ev: TensorNumeric[T]): QuantizedTensor[T] = new QuantizedTensor[T](d1, d2, d3)

  def apply[@specialized(Float, Double) T: ClassTag](d1: Int, d2: Int, d3: Int, d4: Int)(
    implicit ev: TensorNumeric[T]): QuantizedTensor[T] = new QuantizedTensor[T](d1, d2, d3, d4)

  def apply[@specialized(Float, Double) T: ClassTag](d1: Int, d2: Int, d3: Int, d4: Int, d5: Int)(
    implicit ev: TensorNumeric[T]): QuantizedTensor[T] = new QuantizedTensor[T](d1, d2, d3, d4, d5)

  def apply[@specialized(Float, Double) T: ClassTag](size: Array[Int])(
    implicit ev: TensorNumeric[T]): QuantizedTensor[T] = new QuantizedTensor[T](size)

  def apply[@specialized(Float, Double) T: ClassTag](src: Tensor[T], descParams: DescParams,
    descType: DescType)(implicit ev: TensorNumeric[T]): QuantizedTensor[T] = {
    val tensor = new QuantizedTensor[T](src.size(), src.stride(), src.nDimension())
    tensor.value = tensor.createInternalStorage(src)
    tensor.desc = Desc.get(descParams, descType, tensor.value, 0,
      tensor.maxOfRow, tensor.minOfRow)
    tensor.params = descParams
    tensor.descType = descType
    tensor
  }

  def apply[@specialized(Float, Double) T: ClassTag](src: Array[Byte], min: Array[T], max: Array[T],
    sum: Array[T], size: Array[Int])(
    implicit ev: TensorNumeric[T]): QuantizedTensor[T] = {
    val tensor = new QuantizedTensor[T](size)
    tensor.value = src
    tensor.maxOfRow = max
    tensor.minOfRow = min
    tensor.sumOfRow = sum
    tensor
  }
}
