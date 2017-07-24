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

package com.intel.analytics.bigdl.dataset


import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import org.apache.commons.lang3.SerializationUtils

import scala.reflect.ClassTag

/**
 * Class that represents the features and labels of a data sample.
 *
 * @tparam T numeric type
 */
abstract class Sample[T: ClassTag] extends Serializable {
  /**
   * First dimension length of index-th feature.
   * This function could be used to sort samples in [[DataSet]].
   *
   * @return
   */
  def featureLength(index: Int): Int

  /**
   * First dimension length of index-th label.
   * This function could be used to find the longest label.
   *
   * @return
   */
  def labelLength(index: Int): Int

  /**
   * Number of tensors in feature
   *
   * @return number of tensors in feature
   */
  def numFeature(): Int

  /**
   * Number of tensors in label
   *
   * @return number of tensors in label
   */
  def numLabel(): Int

  /**
   *@return A deep clone
   */
  override def clone(): this.type =
    SerializationUtils.clone(this)

  /**
   * Get feature tensor, for one feature Sample only.
   * You don't need to override this, because we have add
   * a default implement to throw exception.
   * @return feature tensor
   */
  @deprecated("Old interface", "0.2.0")
  def feature()(implicit ev: TensorNumeric[T]): Tensor[T] = {
    throw new UnsupportedOperationException("Sample.feature(): unimplemented deprecated method")
  }

  /**
   * Get label tensor, for one label Sample only.
   * You don't need to override this, because we have add
   * a default implement to throw exception.
   * @return label tensor
   */
  @deprecated("Old interface", "0.2.0")
  def label()(implicit ev: TensorNumeric[T]): Tensor[T] = {
    throw new UnsupportedOperationException("Sample.label(): unimplemented deprecated method")
  }

  /**
   * Set data of feature and label.
   * @param featureData
   * @param labelData
   * @param featureSize
   * @param labelSize
   * @return
   */
  @deprecated("Old interface", "0.2.0")
  def set(
        featureData: Array[T],
        labelData: Array[T],
        featureSize: Array[Int],
        labelSize: Array[Int])(implicit ev: TensorNumeric[T]): Sample[T]

  /**
   * Get feature sizes
   * @return feature sizes
   */
  def getFeatureSize(): Array[Array[Int]]


  /**
   * Get label sizes
   * @return label sizes
   */
  def getLabelSize(): Array[Array[Int]]

  /**
   * Get data
   * @return data
   */
  def getData(): Array[T]
}


/**
 * A kind of sample who use only one array
 */
private[bigdl] class ArraySample[T: ClassTag](
      private val data: Array[T],
      private val featureSize: Array[Array[Int]],
      private val labelSize: Array[Array[Int]]) extends Sample[T] {
  require(data != null, "Sample: Data couldn't be empty")
  require(featureSize != null, "Sample: Feature couldn't be empty")

  override def getData(): Array[T] = data

  override def featureLength(index: Int): Int = {
    require(null != featureSize, "featureSize is empty")
    featureSize(index)(0)
  }

  override def labelLength(index: Int): Int = {
    if (null != labelSize) {
      labelSize(index)(0)
    } else {
      0
    }
  }

  override def getFeatureSize(): Array[Array[Int]] = {
    featureSize
  }

  override def getLabelSize(): Array[Array[Int]] = {
    require(null != labelSize, "Sample doesn't have label")
    labelSize
  }

  override def numFeature(): Int = {
    require(null != featureSize, "featureSize is empty")
    featureSize.length
  }

  override def numLabel(): Int = {
    if (null == labelSize) {
      0
    } else {
      labelSize.length
    }
  }

  @deprecated("Old interface", "0.2.0")
  override def feature()(implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(featureSize.length == 1, "Old interface for 1 feature Sample. " +
      s"got ${featureSize.length} feature Sample")
    Tensor[T](Storage(data), 1, getFeatureSize()(0))
  }

  @deprecated("Old interface", "0.2.0")
  override def label()(implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(labelSize.length == 1, "Old interface for 1 label Sample. " +
      s"got ${labelSize.length} label Sample")
    Tensor[T](Storage(data), getFeatureSize().map(_.product).sum + 1,
      labelSize(0))
  }

  @deprecated("Old interface", "0.2.0")
  override def set(
           featureData: Array[T],
           labelData: Array[T],
           featureSize: Array[Int],
           labelSize: Array[Int])(implicit ev: TensorNumeric[T]): Sample[T] = {
    require(featureSize.sameElements(this.featureSize(0)) &&
      labelSize.sameElements(this.labelSize(0)), "size not match")

    ev.arraycopy(featureData, 0, data, 0, featureData.length)
    ev.arraycopy(labelData, 0, data, featureData.length, labelData.length)

    this
  }

  def canEqual(other: Any): Boolean = other.isInstanceOf[ArraySample[T]]

  override def equals(other: Any): Boolean = other match {
    case that: ArraySample[T] =>
      if (!(that canEqual this) ||
        !(data.deep == that.data.deep) ||
        !(featureSize.deep == that.featureSize.deep)) {
        return false
      }
      if (null != labelSize && null != that.labelSize) {
        labelSize.deep == that.labelSize.deep
      } else {
        null == labelSize & null == that.labelSize
      }
    case _ => false
  }

  override def hashCode(): Int = {
    val state = if (null == labelSize) Seq(data, featureSize) else Seq(data, featureSize, labelSize)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }
}

object Sample {
  def apply[T: ClassTag](
      data: Array[T],
      featureSize: Array[Array[Int]],
      labelSize: Array[Array[Int]]): Sample[T] = {
    new ArraySample(data, featureSize, labelSize)
  }

  def apply[T: ClassTag](
        featureTensor: Tensor[T],
        labelTensor: Tensor[T])(implicit ev: TensorNumeric[T]) : Sample[T] = {
    require(featureTensor.isContiguous(), "featureTensor is not contiguous")
    require(labelTensor.isContiguous(), "labelTensor is not contiguous")
    val data = new Array[T](featureTensor.nElement() + labelTensor.nElement())
    ev.arraycopy(featureTensor.storage().array(), featureTensor.storageOffset() - 1,
      data, 0, featureTensor.nElement())
    ev.arraycopy(labelTensor.storage().array(), labelTensor.storageOffset() - 1,
      data, featureTensor.nElement(), labelTensor.nElement())
    new ArraySample[T](data, getSize(featureTensor), getSize(labelTensor))
  }

  def apply[T: ClassTag](
        featureTensor: Tensor[T],
        label: T)(implicit ev: TensorNumeric[T]) : Sample[T] = {
    require(featureTensor.isContiguous(), "featureTensor is not contiguous")
    val data = new Array[T](featureTensor.nElement() + 1)
    ev.arraycopy(featureTensor.storage().array(), featureTensor.storageOffset() - 1,
      data, 0, featureTensor.nElement())
    data(featureTensor.nElement()) = label
    new ArraySample[T](data, getSize(featureTensor), Array(Array(1)))
  }

  def apply[T: ClassTag](
        featureTensors: Array[Tensor[T]],
        labelTensor: Tensor[T])(implicit ev: TensorNumeric[T]) : Sample[T] = {
    val tensors = featureTensors ++ Array(labelTensor)
    val data = new Array[T](tensors.map(_.nElement()).sum)
    copy(data, tensors)
    new ArraySample[T](data, getSize(featureTensors), getSize(labelTensor))
  }

  def apply[T: ClassTag](
        featureTensors: Array[Tensor[T]],
        labelTensors: Array[Tensor[T]])(implicit ev: TensorNumeric[T]) : Sample[T] = {
    val tensors = featureTensors ++ labelTensors
    val data = new Array[T](tensors.map(_.nElement()).sum)
    copy(data, tensors)
    new ArraySample[T](data, getSize(featureTensors), getSize(labelTensors))
  }

  def apply[T: ClassTag](
        featureTensor: Tensor[T])(implicit ev: TensorNumeric[T]) : Sample[T] = {
    require(featureTensor.isContiguous(), "featureTensor is not contiguous")
    val data = new Array[T](featureTensor.nElement())
    ev.arraycopy(featureTensor.storage().array(), featureTensor.storageOffset() - 1,
      data, 0, featureTensor.nElement())
    new ArraySample[T](data, getSize(featureTensor), null)
  }

  def apply[T: ClassTag](
        featureTensors: Array[Tensor[T]])(implicit ev: TensorNumeric[T]) : Sample[T] = {
    val data = new Array[T](featureTensors.map(_.nElement()).sum)
    copy(data, featureTensors)
    new ArraySample[T](featureTensors.flatMap(_.storage().array()),
      getSize(featureTensors), null)
  }

  private def copy[T: ClassTag](
      data: Array[T],
      tensors: Array[Tensor[T]])(implicit ev: TensorNumeric[T]) : Array[T] = {
    var offset = 0
    var i = 0
    while (i < tensors.length) {
      val tensor = tensors(i)
      require(tensor.isContiguous(), s"${i}-th tensor is not contiguous")
      ev.arraycopy(tensor.storage().array(), tensor.storageOffset() - 1,
        data, offset, tensor.nElement())
      offset += tensor.nElement()
      i += 1
    }
    data
  }

  private[bigdl] def getSize[T: ClassTag](tensors: Array[Tensor[T]]): Array[Array[Int]] = {
    tensors.map(_.size)
  }

  private[bigdl] def getSize[T: ClassTag](tensor: Tensor[T]): Array[Array[Int]] = {
    Array(tensor.size())
  }

  private[bigdl] def sameSize(a: Array[Array[Int]], b: Array[Array[Int]]): Boolean = {
    if (a.length != b.length) return false
    var i = 0
    while (i < a.length) {
      if (a(i).length != b(i).length) return false
      i += 1
    }
    true
  }
}
