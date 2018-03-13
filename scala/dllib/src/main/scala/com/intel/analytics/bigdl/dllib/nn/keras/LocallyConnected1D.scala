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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, DataFormat}
import com.intel.analytics.bigdl.nn.{Squeeze, Sequential => TSequential}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape

import scala.reflect.ClassTag

/**
 * Locally-connected layer for 1D inputs which works similarly to the TemporalConvolution layer,
 * except that weights are unshared, that is, a different set of filters
 * is applied at each different patch of the input.
 * Border mode currently supported for this layer is 'valid'.
 * The input of this layer should be 3D.
 *
 * When using this layer as the first layer in a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param nbFilter Dimensionality of the output.
 * @param filterLength The extension (spatial or temporal) of each filter.
 * @param activation Activation function to use. Default is null.
 *                   You can also pass in corresponding string representations such as 'relu'
 *                   or 'sigmoid', etc. for simple activations in the factory method.
 * @param subsampleLength Integer. Factor by which to subsample output.
 * @param wRegularizer An instance of [[Regularizer]], (eg. L1 or L2 regularization),
 *                     applied to the input weights matrices. Default is null.
 * @param bRegularizer An instance of [[Regularizer]], applied to the bias. Default is null.
 * @param bias Whether to include a bias (i.e. make the layer affine rather than linear).
 *             Default is true.
 * @tparam T The numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class LocallyConnected1D[T: ClassTag](
   val nbFilter: Int,
   val filterLength: Int,
   val activation: KerasLayer[Tensor[T], Tensor[T], T] = null,
   val subsampleLength: Int = 1,
   var wRegularizer: Regularizer[T] = null,
   var bRegularizer: Regularizer[T] = null,
   val bias: Boolean = true,
   val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasLayer.addBatch(inputShape)) {

  override def computeOutputShape(inputShape: Shape): Shape = {
    val input = inputShape.toSingle().toArray
    require(input.length == 3,
      s"LocallyConnected1D requires 3D input, but got input dim ${input.length}")
    val length = KerasUtils.computeConvOutputLength(input(1), filterLength,
      "valid", subsampleLength)
    Shape(input(0), length, nbFilter)
  }

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val input = inputShape.toSingle().toArray
    val model = TSequential[T]()
    model.add(com.intel.analytics.bigdl.nn.Reshape(Array(input(1), 1, input(2)), Some(true)))
    val layer = com.intel.analytics.bigdl.nn.LocallyConnected2D(
      nInputPlane = input(2),
      inputWidth = 1,
      inputHeight = input(1),
      nOutputPlane = nbFilter,
      kernelW = 1,
      kernelH = filterLength,
      strideW = 1,
      strideH = subsampleLength,
      wRegularizer = wRegularizer,
      bRegularizer = bRegularizer,
      withBias = bias,
      format = DataFormat.NHWC)
    model.add(layer)
    model.add(Squeeze(3))
    if (activation != null) {
      model.add(activation.doBuild(inputShape))
    }
    model.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object LocallyConnected1D {
  def apply[@specialized(Float, Double) T: ClassTag](
    nbFilter: Int,
    filterLength: Int,
    activation: String = null,
    subsampleLength: Int = 1,
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    bias: Boolean = true,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): LocallyConnected1D[T] = {
    new LocallyConnected1D[T](nbFilter, filterLength,
      KerasUtils.getKerasActivation(activation), subsampleLength,
      wRegularizer, bRegularizer, bias, inputShape)
  }
}
