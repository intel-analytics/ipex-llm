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

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.{Sequential => TSequential}
import com.intel.analytics.bigdl.nn.abstractnn._
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape

import scala.reflect.ClassTag

/**
 * Applies convolution operator for filtering neighborhoods of 1-D inputs.
 * You can also use Conv1D as an alias of this layer.
 * The input of this layer should be 3D.
 *
 * When you use this layer as the first layer of a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param nbFilter Number of convolution filters to use.
 * @param filterLength The extension (spatial or temporal) of each filter.
 * @param init Initialization method for the weights of the layer. Default is Xavier.
 *             You can also pass in corresponding string representations such as 'glorot_uniform'
 *             or 'normal', etc. for simple init methods in the factory method.
 * @param activation Activation function to use. Default is null.
 *                   You can also pass in corresponding string representations such as 'relu'
 *                   or 'sigmoid', etc. for simple activations in the factory method.
 * @param borderMode Either 'valid' or 'same'. Default is 'valid'.
 * @param subsampleLength Factor by which to subsample output. Integer. Default is 1.
 * @param wRegularizer An instance of [[Regularizer]], (eg. L1 or L2 regularization),
 *                     applied to the input weights matrices. Default is null.
 * @param bRegularizer An instance of [[Regularizer]], applied to the bias. Default is null.
 * @param bias Whether to include a bias (i.e. make the layer affine rather than linear).
 *             Default is true.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class Convolution1D[T: ClassTag](
   val nbFilter: Int,
   val filterLength: Int,
   val init: InitializationMethod = Xavier,
   val activation: KerasLayer[Tensor[T], Tensor[T], T] = null,
   val borderMode: String = "valid",
   val subsampleLength: Int = 1,
   var wRegularizer: Regularizer[T] = null,
   var bRegularizer: Regularizer[T] = null,
   val bias: Boolean = true,
   val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasLayer.addBatch(inputShape)) {

  override def computeOutputShape(inputShape: Shape): Shape = {
    val input = inputShape.toSingle().toArray
    require(input.length == 3,
      s"Convolution1D requires 3D input, but got input dim ${input.length}")
    val outputLength = KerasUtils.computeConvOutputLength(input(1), filterLength,
      borderMode, subsampleLength)
    Shape(input(0), outputLength, nbFilter)
  }

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val input = inputShape.toSingle().toArray
    val pads = KerasUtils.getPadsFromBorderMode(borderMode)
    val model = TSequential[T]()
    model.add(com.intel.analytics.bigdl.nn.Reshape(Array(input(1), 1, input(2)), Some(true)))
    val layer = SpatialConvolution(
      nInputPlane = input(2),
      nOutputPlane = nbFilter,
      kernelW = 1,
      kernelH = filterLength,
      strideW = 1,
      strideH = subsampleLength,
      padW = pads._2,
      padH = pads._1,
      wRegularizer = wRegularizer,
      bRegularizer = bRegularizer,
      withBias = bias,
      format = DataFormat.NHWC)
    layer.setInitMethod(weightInitMethod = init, biasInitMethod = Zeros)
    model.add(layer)
    model.add(Squeeze(3))
    if (activation != null) {
      model.add(activation.doBuild(inputShape))
    }
    model.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object Convolution1D {
  def apply[@specialized(Float, Double) T: ClassTag](
    nbFilter: Int,
    filterLength: Int,
    init: String = "glorot_uniform",
    activation: String = null,
    borderMode: String = "valid",
    subsampleLength: Int = 1,
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    bias: Boolean = true,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): Convolution1D[T] = {
    new Convolution1D[T](nbFilter, filterLength,
      KerasUtils.getInitMethod(init), KerasUtils.getKerasActivation(activation),
      borderMode, subsampleLength, wRegularizer, bRegularizer, bias, inputShape)
  }
}
