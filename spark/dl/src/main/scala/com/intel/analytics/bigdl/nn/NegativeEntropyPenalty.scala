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
package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * Penalize the input multinomial distribution if it has low entropy.
 * The input to this layer should be a batch of vector each representing a
 * multinomial distribution. The input is typically the output of a softmax layer.
 *
 * For forward, the output is the same as input and a NegativeEntropy loss of
 * the latent state will be calculated each time. For backward,
 * gradInput = gradOutput + gradLoss
 *
 * This can be used in reinforcement learning to discourage the policy from
 * collapsing to a single action for a given state, which improves exploration.
 * See the A3C paper for more detail (https://arxiv.org/pdf/1602.01783.pdf).
 *
 * @param beta penalty coefficient
 */
@SerialVersionUID(- 5766252125245927237L)
class NegativeEntropyPenalty[T: ClassTag]
(val beta: Double = 0.01)
(implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  var loss: T = ev.fromType(0)
  private val buffer = Tensor[T]()

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    loss = ev.times(buffer.resizeAs(input)
      .copy(input).log().cmul(input).sum(), ev.fromType(beta))
    output = input
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {

    gradInput.resizeAs(input).copy(input)
      .log().add(ev.fromType(1)).mul(ev.fromType(beta))

    gradInput.add(gradOutput)
    gradInput
  }
}

object NegativeEntropyPenalty {
  def apply[@specialized(Float, Double) T: ClassTag](beta: Double = 0.01)
         (implicit ev: TensorNumeric[T]) : NegativeEntropyPenalty[T] = {
    new NegativeEntropyPenalty[T](beta)
  }
}
