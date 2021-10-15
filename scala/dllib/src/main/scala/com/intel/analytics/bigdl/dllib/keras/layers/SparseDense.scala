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
import com.intel.analytics.bigdl.dllib.nn.keras.KerasLayer
import com.intel.analytics.bigdl.dllib.nn.{InferReshape, InitializationMethod, SparseLinear, Xavier, Zeros, Sequential => TSequential}
import com.intel.analytics.bigdl.dllib.optim.Regularizer
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.Net
import com.intel.analytics.bigdl.dllib.keras.layers.utils.KerasUtils

import scala.reflect.ClassTag

/**
 * SparseDense is the sparse version of layer Dense. SparseDense has two different from Dense:
 * firstly, SparseDense's input Tensor is a SparseTensor. Secondly, SparseDense doesn't backward
 * gradient to next layer in the backpropagation by default, as the gradInput of SparseDense is
 * useless and very big in most cases.
 *
 * But, considering model like Wide&Deep, we provide backwardStart and backwardLength to backward
 * part of the gradient to next layer.
 *
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
 * @param backwardStart backwardStart index, counting from 1.
 * @param backwardLength backward length.
 * @param bias Whether to include a bias (i.e. make the layer affine rather than linear).
 *             Default is true.
 * @param inputShape A Single Shape, does not include the batch dimension.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class SparseDense[T: ClassTag](
    val outputDim: Int,
    val init: InitializationMethod = Xavier,
    val activation: KerasLayer[Tensor[T], Tensor[T], T] = null,
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    val backwardStart: Int = -1,
    val backwardLength: Int = -1,
    initWeight: Tensor[T] = null,
    initBias: Tensor[T] = null,
    initGradWeight: Tensor[T] = null,
    initGradBias: Tensor[T] = null,
    val bias: Boolean = true,
    val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasUtils.addBatch(inputShape))
    with Net {

  override def computeOutputShape(inputShape: Shape): Shape = {
    val input = inputShape.toSingle().toArray
    require(inputShape.toSingle().size >=2,
      s"SparseDense requires input dim >=2, but got dim: ${inputShape.toSingle().length}")
    Shape(input.slice(0, input.length -1) ++ Array(outputDim))
  }

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val inputShapeList = inputShape.toSingle()
    val layer = SparseLinear[T](inputSize = inputShape.toSingle().last,
      outputSize = outputDim, withBias = bias, wRegularizer = wRegularizer,
      bRegularizer = bRegularizer, backwardStart = backwardStart,
      backwardLength = backwardLength, initWeight = initWeight,
      initBias = initBias, initGradWeight = initGradWeight, initGradBias = initGradBias)
    layer.setInitMethod(weightInitMethod = init, biasInitMethod = Zeros)

    var torchLayer: AbstractModule[Tensor[T], Tensor[T], T] = layer

  KerasUtils.fuse(torchLayer, activation,
      inputShape).asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object SparseDense {
  def apply[@specialized(Float, Double) T: ClassTag](
      outputDim: Int,
      init: String = "glorot_uniform",
      activation: String = null,
      wRegularizer: Regularizer[T] = null,
      bRegularizer: Regularizer[T] = null,
      backwardStart: Int = -1,
      backwardLength: Int = -1,
      initWeight: Tensor[T] = null,
      initBias: Tensor[T] = null,
      initGradWeight: Tensor[T] = null,
      initGradBias: Tensor[T] = null,
      bias: Boolean = true,
      inputShape: Shape = null)(implicit ev: TensorNumeric[T]): SparseDense[T] = {
    new SparseDense[T](outputDim, KerasUtils.getInitMethod(init),
      KerasUtils.getKerasActivation(activation),
      wRegularizer, bRegularizer, backwardStart, backwardLength,
      initWeight, initBias, initGradWeight, initGradBias, bias, inputShape)
  }
}


