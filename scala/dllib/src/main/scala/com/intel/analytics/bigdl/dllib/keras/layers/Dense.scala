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

package com.intel.analytics.zoo.pipeline.api.keras.layers

import com.intel.analytics.bigdl.nn.keras.{KerasLayer, Dense => BigDLDense}
import com.intel.analytics.bigdl.nn.{InitializationMethod, Xavier}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils

import scala.reflect.ClassTag

/**
 * A densely-connected NN layer.
 * The most common input is 2D.
 *
 * When you use this layer as the first layer of a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param outputDim The size of output dimension.
 * @param init Initialization method for the weights of the layer. Default is Xavier.
 *             You can also pass in corresponding string representations such as 'glorot_uniform'
 *             or 'normal', etc. for simple init methods in the factory method.
 * @param activation Activation function to use. Default is null.
 *                   You can also pass in corresponding string representations such as 'relu'
 *                   or 'sigmoid', etc. for simple activations in the factory method.
 * @param wRegularizer An instance of [[Regularizer]], (eg. L1 or L2 regularization),
 *                     applied to the input weights matrices. Default is null.
 * @param bRegularizer An instance of [[Regularizer]], applied to the bias. Default is null.
 * @param bias Whether to include a bias (i.e. make the layer affine rather than linear).
 *             Default is true.
 * @param inputShape A Single Shape, does not include the batch dimension.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class Dense[T: ClassTag](
  override val outputDim: Int,
  override val init: InitializationMethod = Xavier,
  override val activation: KerasLayer[Tensor[T], Tensor[T], T] = null,
  wRegularizer: Regularizer[T] = null,
  bRegularizer: Regularizer[T] = null,
  override val bias: Boolean = true,
  override val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends BigDLDense[T](outputDim, init, activation, wRegularizer, bRegularizer, bias,
    inputShape) with Net {

  override private[zoo] def toKeras2(): String = {
    val params = Net.inputShapeToString(inputShape) ++
      Net.activationToString(activation) ++
      Net.param(getName()) ++
      Net.param(bias, "use_bias") ++
      Net.param(outputDim, "units")
    Net.kerasDef(this, params)
  }

  override private[zoo] def getKerasWeights(): Array[Tensor[Float]] = {
    val weights = this.parameters()._1
    val kWeights = Array.tabulate(weights.length)(_ => Tensor[Float]())
    weights(0) = weights(0).t().contiguous()
    weights(0).cast[Float](kWeights(0).resizeAs(weights(0)))
    weights(1).cast[Float](kWeights(1).resizeAs(weights(1)))
    kWeights
  }
}

object Dense {
  def apply[@specialized(Float, Double) T: ClassTag](
    outputDim: Int,
    init: String = "glorot_uniform",
    limits: Array[Double] = null,
    activation: String = null,
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    bias: Boolean = true,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): Dense[T] = {
    val initValue = KerasUtils.getInitMethod(init, limits)
    new Dense[T](outputDim, initValue,
      KerasUtils.getKerasActivation(activation),
      wRegularizer, bRegularizer, bias, inputShape)
  }
}
