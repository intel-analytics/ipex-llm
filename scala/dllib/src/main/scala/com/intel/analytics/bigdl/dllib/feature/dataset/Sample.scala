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
import com.intel.analytics.bigdl.tensor.Tensor
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

object Sample {
  def apply[@specialized(Float, Double) T: ClassTag]
  (featureTensor: Tensor[T], labelTensor: Tensor[T])
  (implicit ev: TensorNumeric[T]) : Sample[T] = {
    new TensorSample[T](featureTensor, labelTensor)
  }
}
