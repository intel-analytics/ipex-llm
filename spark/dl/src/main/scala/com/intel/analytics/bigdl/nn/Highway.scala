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
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag


object Highway {
  /**
   * Densely connected highway network.
   * Highway layers are a natural extension of LSTMs to feedforward networks.
   *
   * @param size input size
   * @param withBias whether to include a bias
   * @param activation activation function
   * @param wRegularizer: instance of [[Regularizer]]
   *                    (eg. L1 or L2 regularization), applied to the input weights matrices.
   * @param bRegularizer: instance of [[Regularizer]]
   *                    applied to the bias.
   */
  def apply[@specialized(Float, Double) T: ClassTag](size: Int, withBias: Boolean = true,
    activation: TensorModule[T] = null,
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null)
    (implicit ev: TensorNumeric[T]): Graph[T] = {
    val input = Input()
    val l1 = Linear(size, size, withBias = withBias, wRegularizer = wRegularizer,
      bRegularizer = bRegularizer).inputs(input)
    val transformWeight = Sigmoid().inputs(l1)
    val negatedGate = AddConstant(1).inputs(Negative().inputs(transformWeight))
    val l2 = Linear(size, size, withBias = withBias, wRegularizer = wRegularizer,
      bRegularizer = bRegularizer).inputs(input)
    val transformed = if (null != activation) activation.inputs(l2) else l2
    val transformedGated = CMulTable().inputs(transformWeight, transformed)
    val identityGate = CMulTable().inputs(negatedGate, input)
    val value = CAddTable().inputs(transformedGated, identityGate)
    Graph(Array(input), Array(value))
  }
}
