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

package com.intel.analytics.bigdl.nn.keras

import com.intel.analytics.bigdl.nn.{InferReshape, InitializationMethod, Linear, Xavier, Zeros, Sequential => TSequential}
import com.intel.analytics.bigdl.nn.abstractnn._
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape

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
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class Dense[T: ClassTag](
   val outputDim: Int,
   val init: InitializationMethod = Xavier,
   val activation: KerasLayer[Tensor[T], Tensor[T], T] = null,
   var wRegularizer: Regularizer[T] = null,
   var bRegularizer: Regularizer[T] = null,
   val bias: Boolean = true,
   val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasLayer.addBatch(inputShape)) {

  override def computeOutputShape(inputShape: Shape): Shape = {
    val input = inputShape.toSingle().toArray
    require(inputShape.toSingle().size >=2,
      s"Dense requires input dim >=2, but got dim: ${inputShape.toSingle().length}")
    Shape(input.slice(0, input.length -1) ++ Array(outputDim))
  }

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val inputShapeList = inputShape.toSingle()
    val layer = Linear(
      inputSize = inputShapeList.last,
      outputSize = outputDim,
      withBias = bias,
      wRegularizer = wRegularizer,
      bRegularizer = bRegularizer)
    layer.setInitMethod(weightInitMethod = init, biasInitMethod = Zeros)

    var torchLayer: AbstractModule[Tensor[T], Tensor[T], T] = layer

    if (inputShape.toSingle().size > 2) {
      val seq = new TSequential[T]()
      val inDim = inputShapeList.last
      seq.add(InferReshape(Array(-1, inDim), false))
      seq.add(layer)
      seq.add(InferReshape(Array(-1) ++
        inputShapeList.slice(1, inputShapeList.size - 1) ++ Array(outputDim), false))
      torchLayer = seq.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
    }
    KerasLayer.fuse(torchLayer, activation,
      inputShape).asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object Dense {
  def apply[@specialized(Float, Double) T: ClassTag](
    outputDim: Int,
    init: String = "glorot_uniform",
    activation: String = null,
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    bias: Boolean = true,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): Dense[T] = {
    new Dense[T](outputDim, KerasUtils.getInitMethod(init),
      KerasUtils.getKerasActivation(activation),
      wRegularizer, bRegularizer, bias, inputShape)
  }
}


