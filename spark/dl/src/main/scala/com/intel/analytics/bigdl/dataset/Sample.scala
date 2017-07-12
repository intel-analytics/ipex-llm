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
 * @tparam T numeric type
 */
abstract class Sample[T: ClassTag] extends Serializable {
  /**
   * Length of the feature.
   * This function could be used to sort samples in [[DataSet]].
   * @return
   */
  def featureLength(): Int

  /**
   * Length of the feature.
   * This function could be used to find the longest label.
   * @return
   */
  def labelLength(): Int

  /**
   * Copy other Sample's data to this Sample
   * @param other Sample to be copied.
   * @return this
   */
  def copy(other: Sample[T]): this.type

  /**
   * Number of tensors in feature
   * @return number of tensors in feature
   */
  def numFeature(): Int

  /**
   * Number of tensors in label
   * @return number of tensors in label
   */
  def numLabel(): Int

  /**
   * Deep clone
   * @return a deep clone
   */
  override def clone(): this.type =
    SerializationUtils.clone(this)

  @deprecated("Old interface", "0.2.0")
  def feature(): Tensor[T] = {
    require(this.isInstanceOf[TensorSample[T]], "Deprecated method, Only support TensorSample.")
    this.asInstanceOf[TensorSample[T]].featureTensor
  }

  @deprecated("Old interface", "0.2.0")
  def label(): Tensor[T] = {
    require(this.isInstanceOf[TensorSample[T]], "Deprecated method, Only support TensorSample.")
    this.asInstanceOf[TensorSample[T]].labelTensor
  }

  @deprecated("Old interface", "0.2.0")
  def set(
        featureData: Array[T],
        labelData: Array[T],
        featureSize: Array[Int],
        labelSize: Array[Int]): Sample[T] = {
    require(this.isInstanceOf[TensorSample[T]], "Deprecated method, Only support TensorSample.")
    val sample = this.asInstanceOf[TensorSample[T]]
    sample.featureTensor.set(Storage[T](featureData), 1, featureSize)
    sample.labelTensor.set(Storage[T](labelData), 1, labelSize)
    sample
  }
}

/**
 * A kind of sample. Feature is a tensor, and label is a tensor too.
 * @param featureTensor feature tensor
 * @param labelTensor label tensor
 * @tparam T numeric type
 */
class TensorSample[T: ClassTag](
      val featureTensor: Tensor[T],
      val labelTensor: Tensor[T]) extends Sample[T] {

  /**
   * The length of first dimension
   * @return The length of first dimension
   */
  override def featureLength(): Int = {
    featureTensor.size(1)
  }

  override def labelLength(): Int = {
    labelTensor.size(1)
  }

  override def copy(other: Sample[T]): this.type = {
    require(other.isInstanceOf[TensorSample[T]], "Sample.copy: sample type not match.")
    val s = other.asInstanceOf[TensorSample[T]]
    featureTensor.resizeAs(s.featureTensor).copy(s.featureTensor)
    labelTensor.resizeAs(s.labelTensor).copy(s.labelTensor)
    this
  }

  def numFeature(): Int = 1

  def numLabel(): Int = 1

  def canEqual(other: Any): Boolean = other.isInstanceOf[TensorSample[T]]

  override def equals(other: Any): Boolean = other match {
    case that: TensorSample[T] =>
      (that canEqual this) &&
        featureTensor == that.featureTensor &&
        labelTensor == that.labelTensor
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(featureTensor, labelTensor)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }
}

/**
 * A kind of sample. Feature is a tensor, and label is a double/float.
 * @param featureTensor feature tensor
 * @param labelValue label
 * @tparam T numeric type
 */
class TensorTSample[T: ClassTag](
    val featureTensor: Tensor[T],
    var labelValue: T) extends Sample[T]{
  override def featureLength(): Int = {
    featureTensor.size(1)
  }

  override def labelLength(): Int = {
    1
  }

  override def copy(other: Sample[T]): this.type = {
    require(other.isInstanceOf[TensorTSample[T]], "Sample.copy: sample type not match.")
    val s = other.asInstanceOf[TensorTSample[T]]
    featureTensor.resizeAs(s.featureTensor).copy(s.featureTensor)
    labelValue = s.labelValue
    this
  }

  def numFeature(): Int = 1

  def numLabel(): Int = 1

  def canEqual(other: Any): Boolean = other.isInstanceOf[TensorTSample[T]]

  override def equals(other: Any): Boolean = other match {
    case that: TensorTSample[T] =>
      (that canEqual this) &&
        featureTensor == that.featureTensor &&
        labelValue == that.labelValue
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(featureTensor, labelValue)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }
}

/**
 * A kind of sample. Feature is an Array[Tensor], and label is a tensor too.
 * @param features array of tensor
 * @param labels label tensor
 * @tparam T numeric type
 */
class ArrayTensorSample[T: ClassTag](
    val features: Array[Tensor[T]],
    val labels: Tensor[T]) extends Sample[T] {
  override def featureLength(): Int = {
    features(0).size(1)
  }

  def featuresLength(): Array[Int] = {
    features.map(_.size(1))
  }

  override def labelLength(): Int = {
    labels.size(1)
  }

  override def copy(other: Sample[T]): this.type = {
    require(other.isInstanceOf[ArrayTensorSample[T]], "Sample.copy: sample type not match.")
    val s = other.asInstanceOf[ArrayTensorSample[T]]
    require(s.features.length == features.length, "Sample.copy: sample type not match.")
    var i = 0
    while (i < features.length) {
      features(i).resizeAs(s.features(i)).copy(s.features(i))
      i += 1
    }
    labels.resizeAs(s.labels).copy(s.labels)
    this
  }

  def numFeature(): Int = features.length

  def numLabel(): Int = 1

  def canEqual(other: Any): Boolean = other.isInstanceOf[ArrayTensorSample[T]]

  override def equals(other: Any): Boolean = other match {
    case that: ArrayTensorSample[T] =>
      (that canEqual this) &&
        features == that.features &&
        labels == that.labels
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(features, labels)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }
}

/**
 * A kind of sample doesn't contain label. Feature is a Tensor.
 * @param featureTensor feature tensor
 * @tparam T numeric type
 */
class UnlabeledTensorSample[T: ClassTag](
      val featureTensor: Tensor[T]) extends Sample[T]{
  override def featureLength(): Int = {
    featureTensor.size(1)
  }

  override def labelLength(): Int = 0

  override def copy(other: Sample[T]): this.type = {
    require(other.isInstanceOf[UnlabeledTensorSample[T]], "Sample.copy: sample type not match.")
    val s = other.asInstanceOf[UnlabeledTensorSample[T]]
    featureTensor.resizeAs(s.featureTensor).copy(s.featureTensor)
    this
  }

  def numFeature(): Int = 1

  def numLabel(): Int = 0

  def canEqual(other: Any): Boolean = other.isInstanceOf[UnlabeledTensorSample[T]]

  override def equals(other: Any): Boolean = other match {
    case that: TensorTSample[T] =>
      (that canEqual this) &&
        featureTensor == that.featureTensor
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(featureTensor)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }
}

/**
 * A kind of sample doesn't contain label. Feature is an Array[Tensor].
 * @param features array of tensor
 * @tparam T numeric type
 */
class UnlabeledArrayTensorSample[T: ClassTag](
      val features: Array[Tensor[T]]) extends Sample[T] {
  override def featureLength(): Int = {
    features(0).size(1)
  }

  override def labelLength(): Int = 0

  override def copy(other: Sample[T]): this.type = {
    require(other.isInstanceOf[UnlabeledArrayTensorSample[T]],
      "Sample.copy: sample type not match.")
    val s = other.asInstanceOf[UnlabeledArrayTensorSample[T]]
    require(s.features.length == features.length, "Sample.copy: sample type not match.")
    var i = 0
    while (i < features.length) {
      features(i).resizeAs(s.features(i)).copy(s.features(i))
      i += 1
    }
    this
  }

  def numFeature(): Int = features.length

  def numLabel(): Int = 0

  def canEqual(other: Any): Boolean = other.isInstanceOf[UnlabeledArrayTensorSample[T]]

  override def equals(other: Any): Boolean = other match {
    case that: UnlabeledArrayTensorSample[T] =>
      (that canEqual this) &&
        features == that.features
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(features)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }
}

object Sample {
  def apply[T: ClassTag](
        featureTensor: Tensor[T],
        labelTensor: Tensor[T])(implicit ev: TensorNumeric[T]) : Sample[T] = {
    new TensorSample[T](featureTensor, labelTensor)
  }

  @deprecated("Old interface", "0.2.0")
  def apply[@specialized(Float, Double) T: ClassTag]()(
        implicit ev: TensorNumeric[T]) : Sample[T] = {
    new TensorSample[T](Tensor[T](), Tensor[T]())
  }

  def apply[T: ClassTag](
        featureTensor: Tensor[T],
        label: T)(implicit ev: TensorNumeric[T]) : Sample[T] = {
    new TensorTSample[T](featureTensor, label)
  }

  def apply[T: ClassTag](
        featureTensors: Array[Tensor[T]],
        labelTensor: Tensor[T])(implicit ev: TensorNumeric[T]) : Sample[T] = {
    new ArrayTensorSample[T](featureTensors, labelTensor)
  }

  def apply[T: ClassTag](
        featureTensor: Tensor[T])(implicit ev: TensorNumeric[T]) : Sample[T] = {
    new UnlabeledTensorSample[T](featureTensor)
  }

  def apply[T: ClassTag](
        featureTensors: Array[Tensor[T]])(implicit ev: TensorNumeric[T]) : Sample[T] = {
    new UnlabeledArrayTensorSample[T](featureTensors)
  }
}
