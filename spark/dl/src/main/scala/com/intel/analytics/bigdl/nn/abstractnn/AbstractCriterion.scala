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

package com.intel.analytics.bigdl.nn.abstractnn

import com.intel.analytics.bigdl.nn.abstractnn.SizeAverageStatus.SizeAverageStatus
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}
import org.apache.commons.lang3.SerializationUtils

import scala.reflect.ClassTag

/**
 * [[TensorCriterion]] is an abstract sub-class of [[AbstractCriterion]], whose
 * input and output type both are [[Tensor]].
 *
 * @tparam T The numeric type in the criterion, usually which are [[Float]] or [[Double]]
 */
abstract class TensorCriterion[T: ClassTag]
(implicit ev: TensorNumeric[T]) extends AbstractCriterion[Tensor[T], Tensor[T], T]

/**
 * [[AbstractCriterion]] is an abstract class the concrete criterion should extend.
 * `Criterion`s are helpful to train a neural network. Given an input and a target,
 * they compute the gradient according to a loss function.
 *
 * It provides some important method such as `forward`, `backward`, `updateOutput`,
 * `updateGradInput` frequently used as a criteria. Some of them need to be override
 * in a concrete criterion class.
 *
 * @tparam A represents the input type of the criterion, which an be abstract type [[Activity]],
 *           or concrete type [[Tensor]] or [[Table]]
 * @tparam B represents the output type of the criterion
 * @tparam T The numeric type in the criterion, usually which are [[Float]] or [[Double]]
 */
abstract class AbstractCriterion[A <: Activity: ClassTag, B <: Activity: ClassTag,
 T: ClassTag](
  implicit ev: TensorNumeric[T]) extends Serializable {
  var gradInput: A = Activity.allocate[A, T]()
  var output: T = ev.fromType[Int](0)

  private[nn] var sizeAverageStatus: SizeAverageStatus = SizeAverageStatus.None

  private[nn] def allocateAs[D <: Activity](dest: D): D = dest match {
    case tensor: Tensor[T] => Tensor[T]().asInstanceOf[D]
    case table: Table => T().asInstanceOf[D]
    case _ => throw new IllegalArgumentException("Activity only support tensor and table now")
  }

  /**
   * Takes an input object, and computes the corresponding loss of the criterion,
   * compared with `target`.
   *
   * @param input input data
   * @param target target
   * @return the loss of criterion
   */
  def forward(input: A, target: B): T = {
    updateOutput(input, target)
  }

  /**
   * Performs a back-propagation step through the criterion, with respect to the given input.
   *
   * @param input input data
   * @param target target
   * @return gradient corresponding to input data
   */
  def backward(input: A, target: B): A = {
    updateGradInput(input, target)
  }

  /**
   * Computes the loss using input and objective function. This function
   * returns the result which is stored in the output field.
   *
   * @param input input of the criterion
   * @param target target or labels
   * @return the loss of the criterion
   */
  def updateOutput(input: A, target: B): T = {
    this.output
  }

  /**
   * Computing the gradient of the criterion with respect to its own input. This is returned in
   * gradInput. Also, the gradInput state variable is updated accordingly.
   *
   * @param input input data
   * @param target target data / labels
   * @return gradient of input
   */
  def updateGradInput(input: A, target: B): A

  /**
   * Deep copy this criterion
   * @return a deep copied criterion
   */
  def cloneCriterion(): AbstractCriterion[A, B, T] = {
    SerializationUtils.clone(this)
  }


  def canEqual(other: Any): Boolean = other.isInstanceOf[AbstractCriterion[A, B, T]]

  override def equals(other: Any): Boolean = other match {
    case that: AbstractCriterion[A, B, T] =>
      (that canEqual this) &&
        (that.getClass equals this.getClass) &&
        output == that.output
    case _ => false
  }

  override def hashCode(): Int = {
    def getHashCode(a: Any): Int = if (a == null) 0 else a.hashCode()
    val state = Seq(output)
    state.map(getHashCode).foldLeft(0)((a, b) => 31 * a + b)
  }
}

object SizeAverageStatus extends Enumeration {
  type SizeAverageStatus = Value
  val True, False, None = Value
}
