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
import com.intel.analytics.bigdl.tensor.{DoubleType, FloatType, Storage, Tensor}
import org.apache.commons.lang3.SerializationUtils

import scala.reflect.ClassTag

/**
 * Class that represents the features and labels of a data sample.
 *
 * @tparam T numeric type
 */
abstract class Sample[T: ClassTag] extends Serializable {
  /**
   * Length of the features.
   * This function could be used to sort samples in [[DataSet]].
   *
   * @return
   */
  def featureLength(): Array[Int]

  /**
   * Length of the labels.
   * This function could be used to find the longest label.
   *
   * @return
   */
  def labelLength(): Array[Int]

  /**
   * Copy other Sample's data to this Sample
   *
   * @param other Sample to be copied.
   * @return this
   */
  def copy(other: Sample[T]): this.type

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
   * Deep clone
   *
   * @return a deep clone
   */
  override def clone(): this.type =
    SerializationUtils.clone(this)

  /**
   * Get feature tensor, for one feature Sample only.
   * @return feature tensor
   */
  @deprecated("Old interface", "0.2.0")
  def feature()(implicit ev: TensorNumeric[T]): Tensor[T] = {
    throw new UnsupportedOperationException("Sample.feature(): unimplemented deprecated method")
  }

  /**
   * Get label tensor, for one label Sample only.
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
   * Set sample's data with a set of features and labels.
   * @param features a set of features
   * @param labels a set of labels
   * @return this
   */
  def set(features: Array[Tensor[T]], labels: Array[Tensor[T]]
         )(implicit ev: TensorNumeric[T]): Sample[T]
}


/**
 * A kind of sample who use only one array
 */
private[bigdl] class ArraySample[T: ClassTag](
      private var data: Array[T],
      private var featureSize: Array[Array[Int]],
      private var labelSize: Array[Array[Int]]) extends Sample[T] {

  def getData(): Array[T] = data

  override def featureLength(): Array[Int] = {
    require(null != featureSize, "featureSize is empty")
    featureSize.map(_.product)
  }

  override def labelLength(): Array[Int] = {
    if (null == labelSize) {
      Array(0)
    } else {
      labelSize.map(_.product)
    }
  }

  def getFeatureSize(): Array[Array[Int]] = {
    featureSize
  }

  def getLabelSize(): Array[Array[Int]] = {
    if (null != labelSize) {
      labelSize
    } else {
      Array(Array(0))
    }
  }

  override def copy(other: Sample[T]): this.type = {
    require(other.isInstanceOf[ArraySample[T]], "Sample.copy: sample type not match.")
    val s = other.asInstanceOf[ArraySample[T]]
    require(
      featureSize.length == s.featureSize.length,
      labelSize.length == s.labelSize.length)
    if (data.length < s.getData().length) {
      data = new Array[T](s.getData().length)
    }
    System.arraycopy(s.data, 0, this.data, 0, s.getData().length)
    System.arraycopy(s.featureSize, 0, this.featureSize, 0, this.featureSize.length)
    System.arraycopy(s.labelSize, 0, this.labelSize, 0, this.labelSize.length)
    this
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

  override def set(
        features: Array[Tensor[T]],
        labels: Array[Tensor[T]])(implicit ev: TensorNumeric[T]): Sample[T] = {
    // Resize
    resize(features.map(_.nElement()).sum + labels.map(_.nElement()).sum,
      features.map(_.size()), labels.map(_.size()))

    // Deep copy
    var dataOffset = 0
    var sizeOffset = 0
    var i = 0
    while(i < features.length) {
      val f = features(i)
      require(f.isContiguous())
      // copy data
      ev.arraycopy(f.storage().array(), f.storageOffset() - 1, data, dataOffset, f.nElement())

      dataOffset += f.nElement()
      sizeOffset += f.dim() + 1
      i += 1
    }
    i = 0
    sizeOffset = 0
    while(i < labels.length) {
      val l = labels(i)
      require(l.isContiguous())
      // Copy data
      ev.arraycopy(l.storage().array(), l.storageOffset() - 1, data, dataOffset, l.nElement())

      dataOffset += l.nElement()
      sizeOffset += l.dim()
      i += 1
    }

    this
  }

  @deprecated("Old interface", "0.2.0")
  override def feature()(implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(featureSize.length == 1)
    Tensor[T](Storage(data), 1, getFeatureSize()(0))
  }

  @deprecated("Old interface", "0.2.0")
  override def label()(implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(labelSize.length == 1)
    Tensor[T](Storage(data), getFeatureSize().map(_.product).sum + 1,
      labelSize(0))
  }

  private def resize(
        dataLength: Int,
        featureSize: Array[Array[Int]],
        labelSize: Array[Array[Int]]): Unit = {
    if (data == null || data.length < dataLength) {
      data = new Array[T](dataLength)
    }
    this.featureSize = featureSize
    this.labelSize = labelSize
  }

  @deprecated("Old interface", "0.2.0")
  override def set(
           featureData: Array[T],
           labelData: Array[T],
           featureSize: Array[Int],
           labelSize: Array[Int])(implicit ev: TensorNumeric[T]): Sample[T] = {
    // Resize
    resize(featureData.length + labelData.length,
      Array(featureSize),
      Array(labelSize))
    ev.arraycopy(featureData, 0, data, 0, featureData.length)
    ev.arraycopy(labelData, 0, data, featureData.length, labelData.length)

    this
  }

  def canEqual(other: Any): Boolean = other.isInstanceOf[ArraySample[T]]

  override def equals(other: Any): Boolean = other match {
    case that: ArraySample[T] =>
      (that canEqual this) &&
        data.sameElements(that.data) &&
        featureSize.sameElements(that.featureSize) &&
        labelSize.sameElements(that.labelSize)
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(data, featureSize, labelSize)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }
}

object Sample {
  def apply[T: ClassTag](
        featureTensor: Tensor[T],
        labelTensor: Tensor[T])(implicit ev: TensorNumeric[T]) : Sample[T] = {
    require(featureTensor.isContiguous() && labelTensor.isContiguous())
    val data = new Array[T](featureTensor.nElement() + labelTensor.nElement())
    ev.arraycopy(featureTensor.storage().array(), featureTensor.storageOffset() - 1,
      data, 0, featureTensor.nElement())
    ev.arraycopy(labelTensor.storage().array(), labelTensor.storageOffset() - 1,
      data, featureTensor.nElement(), labelTensor.nElement())
    new ArraySample[T](data, getSize(featureTensor), getSize(labelTensor))
  }

  def apply[@specialized(Float, Double) T: ClassTag]()(
        implicit ev: TensorNumeric[T]) : Sample[T] = {
    new ArraySample[T](Array[T](), null, null)
  }

  def apply[T: ClassTag](
        featureTensor: Tensor[T],
        label: T)(implicit ev: TensorNumeric[T]) : Sample[T] = {
    new ArraySample[T](featureTensor.storage().array ++ Array(label),
      getSize(featureTensor), Array(Array(1)))
  }

  def apply[T: ClassTag](
        featureTensors: Array[Tensor[T]],
        labelTensor: Tensor[T])(implicit ev: TensorNumeric[T]) : Sample[T] = {
    new ArraySample[T]((featureTensors ++ Array(labelTensor)).flatMap(_.storage().array),
      getSize(featureTensors), getSize(labelTensor))
  }

  def apply[T: ClassTag](
        featureTensors: Array[Tensor[T]],
        labelTensors: Array[Tensor[T]])(implicit ev: TensorNumeric[T]) : Sample[T] = {
    new ArraySample[T]((featureTensors ++ labelTensors).filter(_.dim() != 0)
      .flatMap(_.storage().array),
      getSize(featureTensors), getSize(labelTensors))
  }

  def apply[T: ClassTag](
        featureTensor: Tensor[T])(implicit ev: TensorNumeric[T]) : Sample[T] = {
    new ArraySample[T](featureTensor.storage().array(),
      getSize(featureTensor), null)
  }

  def apply[T: ClassTag](
        featureTensors: Array[Tensor[T]])(implicit ev: TensorNumeric[T]) : Sample[T] = {
    new ArraySample[T](featureTensors.flatMap(_.storage().array()),
      getSize(featureTensors), null)
  }

  private[bigdl] def getSize[T: ClassTag](tensors: Array[Tensor[T]]): Array[Array[Int]] = {
    tensors.map(_.size)
  }

  private[bigdl] def getSize[T: ClassTag](tensor: Tensor[T]): Array[Array[Int]] = {
    Array(tensor.size())
  }
}
