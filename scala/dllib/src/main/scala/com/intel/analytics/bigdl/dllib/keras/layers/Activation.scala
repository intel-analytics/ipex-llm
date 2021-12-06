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

package com.intel.analytics.bigdl.dllib.keras.layers

import com.intel.analytics.bigdl.dllib.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.dllib.nn.internal.{Activation => BigDLActivation}
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.Shape
// import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.bigdl.dllib.keras.Net
import com.intel.analytics.bigdl.dllib.keras.layers.utils.KerasUtils

import scala.reflect.ClassTag

/**
 * Simple activation function to be applied to the output.
 * Available activations: 'tanh', 'relu', 'sigmoid', 'softmax', 'softplus', 'softsign',
 *                        'hard_sigmoid', 'linear', 'relu6', 'tanh_shrink', 'softmin',
 *                        'log_sigmoid' and 'log_softmax'.
 *
 * When you use this layer as the first layer of a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param activation Name of the activation function as string.
 * @param inputShape A Single Shape, does not include the batch dimension.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class Activation[T: ClassTag](
    override val activation: String,
    override val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends BigDLActivation[T](activation, inputShape) with Net {

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val kerasActivation = KerasUtils.getKerasActivation(activation)
    kerasActivation.doBuild(inputShape)
  }
}

object Activation {
  def apply[@specialized(Float, Double) T: ClassTag](
    activation: String,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): Activation[T] = {
    new Activation[T](activation, inputShape)
  }
}
