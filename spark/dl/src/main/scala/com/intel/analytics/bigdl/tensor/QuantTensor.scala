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
import com.intel.analytics.bigdl.nn.bigquant.{Desc, DescParams, DescType, Quant}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table
import java.nio.ByteBuffer
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Matrix
import scala.reflect.ClassTag

@SerialVersionUID(- 1766499387282335147L)
private[bigdl] class QuantTensor[@specialized(Float) T: ClassTag](
  private[bigdl] var _size: Array[Int],
  private[bigdl] var _stride: Array[Int],
  var nDimension: Int)(implicit ev: TensorNumeric[T]) extends QuantTensorUnsupported[T] {
  @transient private var desc = 0L
  private var interStorage: Array[Byte] = null
  private var setFromOther: Boolean = false

  var maxOfRow: Array[T] = null
  var minOfRow: Array[T] = null
  var sumOfRow: Array[T] = null

  var params: DescParams = _
  var descType: DescType = _

  private[bigdl] def setStorage(buffer: Array[Byte]): this.type = {
    interStorage = buffer
    this
  }

  def getStorage: Array[Byte] = {
    interStorage
  }

  def getNativeStorage: Long = {
    desc
  }

  def release(): Unit = {
    if (desc != 0 && !setFromOther) {
      BigQuant.FreeMemory(desc)
    }
    desc = 0L
  }

  override def equals(obj: Any): Boolean = {
    if (!super.equals(obj)) {
      return false
    }
    val other = obj.asInstanceOf[QuantTensor[T]]
    if (this.eq(other)) {
      return true
    }

    desc == other.desc
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

    hash = hash * seed + desc.hashCode()

    if (interStorage != null) {
      var i = 0
      while (i < interStorage.length) {
        hash = hash * seed + interStorage(i).toFloat.hashCode()
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

  private def createInterStorage(tensor: Tensor[T]): Array[Byte] = {
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
        Quant.quantize(tensor.asInstanceOf[Tensor[Float]], bytes, bytesOffset)
      case _ =>
        throw new UnsupportedOperationException(s"Only support Float for quantized model")
    }

    bytes
  }

  // rename init
  def init(params: DescParams, descType: DescType): Unit = {
    if (desc != 0L) { release() }

    this.params = params
    this.descType = descType

    this.desc = Desc.get(params, descType, this)
  }

  override def getTensorType: TensorType = QuantType

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
    release()

    interStorage = null
    this
  }

  override def set(other: Tensor[T]): Tensor[T] = {
    other match {
      case quantizedTensor: QuantTensor[T] =>
        if (!this.eq(quantizedTensor)) {
          release() // release first, otherwise will leak memory

          desc = quantizedTensor.getNativeStorage
          interStorage = quantizedTensor.getStorage

          setFromOther = true
        }
      case _ =>
        throw new UnsupportedOperationException(errorString)
    }

    this
  }

  override def copy(other: Tensor[T]): Tensor[T] = {
    if (other.isInstanceOf[QuantTensor[T]]) {
      val o = other.asInstanceOf[QuantTensor[T]]
      this.desc = o.getNativeStorage
      this.interStorage = o.getStorage
    }
    this
  }
}

object QuantTensor {
  /**
   * Returns an empty tensor.
   *
   * @param ev
   * @tparam T
   * @return
   */
  def apply[@specialized(Float, Double) T: ClassTag]()(
    implicit ev: TensorNumeric[T]): QuantTensor[T] = new QuantTensor[T]()
  /**
   * Create a tensor up to 5 dimensions. The tensor size will be `d1 x d2 x d3 x d4 x d5`.
   *
   * @param d1,(d2, d3, d4, d5)
   * @param ev
   * @tparam T
   * @return
   */
  def apply[@specialized(Float, Double) T: ClassTag](d1: Int)(
    implicit ev: TensorNumeric[T]): QuantTensor[T] = new QuantTensor[T](d1)

  def apply[@specialized(Float, Double) T: ClassTag](d1: Int, d2: Int)(
    implicit ev: TensorNumeric[T]): QuantTensor[T] = new QuantTensor[T](d1, d2)

  def apply[@specialized(Float, Double) T: ClassTag](d1: Int, d2: Int, d3: Int)(
    implicit ev: TensorNumeric[T]): QuantTensor[T] = new QuantTensor[T](d1, d2, d3)

  def apply[@specialized(Float, Double) T: ClassTag](d1: Int, d2: Int, d3: Int, d4: Int)(
    implicit ev: TensorNumeric[T]): QuantTensor[T] = new QuantTensor[T](d1, d2, d3, d4)

  def apply[@specialized(Float, Double) T: ClassTag](d1: Int, d2: Int, d3: Int, d4: Int, d5: Int)(
    implicit ev: TensorNumeric[T]): QuantTensor[T] = new QuantTensor[T](d1, d2, d3, d4, d5)

  def apply[@specialized(Float, Double) T: ClassTag](size: Array[Int])(
    implicit ev: TensorNumeric[T]): QuantTensor[T] = new QuantTensor[T](size)

  def apply[@specialized(Float, Double) T: ClassTag](src: Tensor[T], descParams: DescParams,
    descType: DescType)(implicit ev: TensorNumeric[T]): QuantTensor[T] = {
    val tensor = new QuantTensor[T](src.size(), src.stride(), src.nDimension())
    tensor.interStorage = tensor.createInterStorage(src)
    tensor.desc = Desc.get(descParams, descType, tensor.interStorage, 0,
      tensor.maxOfRow, tensor.minOfRow)
    tensor.params = descParams
    tensor.descType = descType
    tensor
  }
}
