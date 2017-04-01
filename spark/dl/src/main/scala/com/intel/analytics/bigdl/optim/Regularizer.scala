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

trait Regularizer[T]
  extends Serializable {
  var isRegualrized: Boolean = true

  def accRegularization(
    parameter: Tensor[T],
    gradParameter: Tensor[T]
  ): Unit
}

@SerialVersionUID(- 5617491971070914067L)
class L1L2Regularizer[T](
  l1: Double,
  l2: Double
)(implicit ev: TensorNumeric[T])
  extends Regularizer[T] {
  override def accRegularization(
    parameter: Tensor[T],
    gradParameter: Tensor[T]
  ): Unit = {
    if (!isRegualrized) return
    accL1L2Regularization(l1, l2, parameter, gradParameter)
  }

  /**
   * Accumulates the gradient of the l1, l2 regularization of `parameter`
   * to `gradParameter`
   *
   * @param l1Alpha l1 regularization rate
   * @param l2Alpha l2 regularization rate
   * @param parameter the parameter that is regularized
   * @param gradParameter the gradient of the parameter
   */
  def accL1L2Regularization(
    l1Alpha: Double,
    l2Alpha: Double,
    parameter: Tensor[T],
    gradParameter: Tensor[T]
  ): Unit = {
    accL1Regularization(l1Alpha, parameter, gradParameter)
    accL2Regularization(l2Alpha, parameter, gradParameter)
  }

  /**
   * Accumulates the gradient of the l1 regularization of `parameter`
   * to `gradParameter`
   *
   * @param alpha l1 regularization rate
   * @param parameter the parameter that is regularized
   * @param gradParameter the gradient of the parameter
   */
  def accL1Regularization(
    alpha: Double,
    parameter: Tensor[T],
    gradParameter: Tensor[T]
  ): Unit = {
    if (alpha != 0) {
      gradParameter.add(ev.fromType(alpha), parameter.sign())
    }
  }

  /**
   * Accumulates the gradient of the l2 regularization of `parameter`
   * to `gradParameter`
   *
   * @param alpha l2 regularization rate
   * @param parameter the parameter that is regularized
   * @param gradParameter the gradient of the parameter
   */
  def accL2Regularization(
    alpha: Double,
    parameter: Tensor[T],
    gradParameter: Tensor[T]
  ): Unit = {
    gradParameter.add(ev.fromType(alpha), parameter)
  }
}

@SerialVersionUID(1950693435414946281L)
class L1Regularizer[T](
  l1: Double
) (implicit ev: TensorNumeric[T])
  extends L1L2Regularizer[T](l1, 0) {
  override def accRegularization(
    parameter: Tensor[T],
    gradParameter: Tensor[T]
  ): Unit = {
    if (!isRegualrized) return
    accL1Regularization(l1, parameter, gradParameter)
  }
}

@SerialVersionUID(- 6597840589687540202L)
case class L2Regularizer[T](
  l2: Double
) (implicit ev: TensorNumeric[T])
  extends L1L2Regularizer[T](0, l2) {
  override def accRegularization(
    parameter: Tensor[T],
    gradParameter: Tensor[T]
  ): Unit = {
    accL2Regularization(l2, parameter, gradParameter)
  }
}
