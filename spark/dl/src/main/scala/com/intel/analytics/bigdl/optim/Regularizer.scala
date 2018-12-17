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

package com.intel.analytics.bigdl.optim

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * It is a trait for all regularizers.
 * Any regularizers need to inherit the result.
 *
 * @tparam T type parameters [[Float]] or [[Double]]
 */
trait Regularizer[T]
  extends Serializable {
  private var isRegualrized: Boolean = true

  /**
   * Enable the regularization feature
   */
  def enable(): Unit = isRegualrized = true

  /**
   * Disable the regularization feature
   */
  def disable(): Unit = isRegualrized = false

  /**
   * The method need to be override by the concrete regularizer class
   * It accumulates the gradient of the regularization of `parameter` to `gradParameter`
   *
   * @param parameter the parameter that is regularized
   * @param gradParameter the gradient of the parameter
   * @param scale the scale of gradParameters
   */
  def accRegularization(
    parameter: Tensor[T],
    gradParameter: Tensor[T],
    scale: Double
  ): Unit

  /**
   * Check the regularization is applied or not
   *
   * @param parameter the parameter that is regularized
   * @param gradParameter the gradient of the parameter
   * @return a boolean, if true, accumulates the gradient of regularization,
   *         otherwise not.
   */
  protected def preCheck(
    parameter: Tensor[T],
    gradParameter: Tensor[T]
  ): Boolean = {
    if (null == parameter
      || null == gradParameter
      || !isRegualrized) {
      false
    } else {
      true
    }
  }
}

/**
 * Apply both L1 and L2 regularization
 * @param l1 l1 regularization rate
 * @param l2 l2 regularization rate
 * @tparam T type parameters [[Float]] or [[Double]]
 */
@SerialVersionUID(- 5617491971070914067L)
class L1L2Regularizer[T: ClassTag](
  val l1: Double,
  val l2: Double
)(implicit ev: TensorNumeric[T])
  extends Regularizer[T] {
  override def accRegularization(
    parameter: Tensor[T],
    gradParameter: Tensor[T],
    scale: Double
  ): Unit = {
    if (!preCheck(parameter, gradParameter)) return
    accL1L2Regularization(l1, l2, parameter, gradParameter, scale)
  }

  /**
   * Accumulates the gradient of the l1, l2 regularization of `parameter`
   * to `gradParameter`
   *
   * @param l1Alpha l1 regularization rate
   * @param l2Alpha l2 regularization rate
   * @param parameter the parameter that is regularized
   * @param gradParameter the gradient of the parameter
   * @param scale scale of gradParameters
   */
  private def accL1L2Regularization(
    l1Alpha: Double,
    l2Alpha: Double,
    parameter: Tensor[T],
    gradParameter: Tensor[T],
    scale: Double
  ): Unit = {
    accL1Regularization(l1Alpha, parameter, gradParameter, scale)
    accL2Regularization(l2Alpha, parameter, gradParameter, scale)
  }

  /**
   * Accumulates the gradient of the l1 regularization of `parameter`
   * to `gradParameter`
   *
   * @param alpha l1 regularization rate
   * @param parameter the parameter that is regularized
   * @param gradParameter the gradient of the parameter
   * @param scale scale of gradParameters
   */
  private def accL1Regularization(
    alpha: Double,
    parameter: Tensor[T],
    gradParameter: Tensor[T],
    scale: Double
  ): Unit = {
    if (alpha != 0 && scale != 0) {
      if (null == l1SignBuffer) l1SignBuffer = Tensor()
      gradParameter.add(ev.fromType(alpha*scale),
        l1SignBuffer.resizeAs(parameter).copy(parameter).sign())
    }
  }

  @transient private var l1SignBuffer: Tensor[T] = null

  /**
   * Accumulates the gradient of the l2 regularization of `parameter`
   * to `gradParameter`
   *
   * @param alpha l2 regularization rate
   * @param parameter the parameter that is regularized
   * @param gradParameter the gradient of the parameter
   * @param scale scale of gradParameters
   */
  private def accL2Regularization(
    alpha: Double,
    parameter: Tensor[T],
    gradParameter: Tensor[T],
    scale: Double
  ): Unit = {
    if (alpha != 0 && scale != 0) gradParameter.add(ev.fromType(alpha* scale), parameter)
  }
}

object L1L2Regularizer {
  def apply[@specialized(Float, Double) T: ClassTag](
    l1: Double,
    l2: Double
  )(implicit ev: TensorNumeric[T]): L1L2Regularizer[T] = new L1L2Regularizer(l1, l2)
}

/**
 * Apply L1 regularization
 * @param l1 l1 regularization rate
 * @tparam T type parameters [[Float]] or [[Double]]
 */
@SerialVersionUID(1950693435414946281L)
case class L1Regularizer[T: ClassTag](
  override val l1: Double
) (implicit ev: TensorNumeric[T])
  extends L1L2Regularizer[T](l1, 0)

/**
 * Apply L2 regularization
 * @param l2 l2 regularization rate
 * @tparam T type parameters [[Float]] or [[Double]]
 */
@SerialVersionUID(- 6597840589687540202L)
case class L2Regularizer[T: ClassTag](
  override val l2: Double
) (implicit ev: TensorNumeric[T])
  extends L1L2Regularizer[T](0, l2)

