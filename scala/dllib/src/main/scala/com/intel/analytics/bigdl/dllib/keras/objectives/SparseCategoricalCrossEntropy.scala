/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.pipeline.api.keras.objectives

import com.intel.analytics.bigdl.nn.abstractnn.TensorCriterion
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.nn.ZooClassNLLCriterion

import scala.reflect.ClassTag

/**
 * A loss often used in multi-class classification problems with SoftMax
 * as the last layer of the neural network.
 *
 * By default, input(y_pred) is supposed to be probabilities of each class,
 * and target(y_true) is supposed to be the class label starting from 0.
 *
 * @param logProbAsInput Boolean. Whether to accept log-probabilities or probabilities
 *                       as input. Default is false and inputs should be probabilities.
 * @param zeroBasedLabel Boolean. Whether target labels start from 0. Default is true.
 *                       If false, labels start from 1.
 * @param weights Tensor. Weights of each class if you have an unbalanced training set.
 *                Default is null.
 * @param sizeAverage Boolean. Whether losses are averaged over observations for each
 *                    mini-batch. Default is true. If false, the losses are instead
 *                    summed for each mini-batch.
 * @param paddingValue Integer. If the target is set to this value, the training process
 *                     will skip this sample. In other words, the forward process will
 *                     return zero output and the backward process will also return
 *                     zero gradInput. Default is -1.
 */
class SparseCategoricalCrossEntropy[T: ClassTag](
    val logProbAsInput: Boolean = false,
    val zeroBasedLabel: Boolean = true,
    val weights: Tensor[T] = null,
    val sizeAverage: Boolean = true,
    val paddingValue: Int = -1)(implicit ev: TensorNumeric[T])
  extends TensorLossFunction[T] {

  override val loss: TensorCriterion[T] =
    ZooClassNLLCriterion[T](weights, sizeAverage, logProbAsInput, paddingValue)

  private val targetBuffer: Tensor[T] = Tensor[T]()

  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    if (zeroBasedLabel) {
      targetBuffer.resizeAs(target)
      targetBuffer.fill(ev.one).add(target)
      super.updateOutput(input, targetBuffer)
    }
    else {
      super.updateOutput(input, target)
    }
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    if (zeroBasedLabel) {
      targetBuffer.resizeAs(target)
      targetBuffer.fill(ev.one).add(target)
      super.updateGradInput(input, targetBuffer)
    }
    else {
      super.updateGradInput(input, target)
    }
  }
}

object SparseCategoricalCrossEntropy {
  def apply[@specialized(Float, Double) T: ClassTag](
      logProbAsInput: Boolean = false,
      zeroBasedLabel: Boolean = true,
      weights: Tensor[T] = null,
      sizeAverage: Boolean = true,
      paddingValue: Int = -1)
    (implicit ev: TensorNumeric[T]): SparseCategoricalCrossEntropy[T] = {
    new SparseCategoricalCrossEntropy[T](logProbAsInput, zeroBasedLabel,
      weights, sizeAverage, paddingValue)
  }
}
