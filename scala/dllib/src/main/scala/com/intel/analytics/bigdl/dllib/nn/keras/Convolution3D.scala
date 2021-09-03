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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule}
import com.intel.analytics.bigdl.nn.{InitializationMethod, VolumetricConvolution, Xavier, Zeros}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape

import scala.reflect.ClassTag

/**
 * Applies convolution operator for filtering windows of three-dimensional inputs.
 * You can also use Conv3D as an alias of this layer.
 * Data format currently supported for this layer is 'CHANNEL_FIRST' (dimOrdering='th').
 * The input of this layer should be 5D.
 *
 * When you use this layer as the first layer of a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension),
 * e.g. inputShape=Shape(3, 10, 128, 128) 10 frames of 128x128 RGB pictures.
 *
 * @param nbFilter Number of convolution filters to use.
 * @param kernelDim1 Length of the first dimension in the convolution kernel.
 * @param kernelDim2 Length of the second dimension in the convolution kernel.
 * @param kernelDim3 Length of the third dimension in the convolution kernel.
 * @param init Initialization method for the weights of the layer. Default is Xavier.
 *             You can also pass in corresponding string representations such as 'glorot_uniform'
 *             or 'normal', etc. for simple init methods in the factory method.
 * @param activation Activation function to use. Default is null.
 *                   You can also pass in corresponding string representations such as 'relu'
 *                   or 'sigmoid', etc. for simple activations in the factory method.
 * @param borderMode Either 'valid' or 'same'. Default is 'valid'.
 * @param subsample Int array of length 3. Factor by which to subsample output.
 *                  Also called strides elsewhere. Default is (1, 1, 1).
 * @param dimOrdering Format of the input data. Please use "CHANNEL_FIRST" (dimOrdering='th').
 * @param wRegularizer An instance of [[Regularizer]], (eg. L1 or L2 regularization),
 *                     applied to the input weights matrices. Default is null.
 * @param bRegularizer An instance of [[Regularizer]], applied to the bias. Default is null.
 * @param bias Whether to include a bias (i.e. make the layer affine rather than linear).
 *             Default is true.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class Convolution3D[T: ClassTag](
   val nbFilter: Int,
   val kernelDim1: Int,
   val kernelDim2: Int,
   val kernelDim3: Int,
   val init: InitializationMethod = Xavier,
   val activation: KerasLayer[Tensor[T], Tensor[T], T] = null,
   val borderMode: String = "valid",
   val subsample: Array[Int] = Array(1, 1, 1),
   val dimOrdering: String = "CHANNEL_FIRST",
   val wRegularizer: Regularizer[T] = null,
   var bRegularizer: Regularizer[T] = null,
   val bias: Boolean = true,
   val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasLayer.addBatch(inputShape)) {

  require(dimOrdering.toLowerCase() == "channel_first", s"Pooling3D currently only supports " +
    s"format CHANNEL_FIRST, but got format $dimOrdering")
  require(borderMode == "valid" || borderMode == "same", s"Invalid border mode for " +
    s"Convolution3D: $borderMode")
  require(subsample.length == 3,
    s"For Convolution3D, subsample should be of length 3 but got length ${subsample.length}")

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val input = inputShape.toSingle().toArray
    val pads = KerasUtils.getPadsFromBorderMode3D(borderMode)
    val layer = VolumetricConvolution(
      nInputPlane = input(1),
      nOutputPlane = nbFilter,
      kT = kernelDim1,
      kW = kernelDim3,
      kH = kernelDim2,
      dT = subsample(0),
      dW = subsample(2),
      dH = subsample(1),
      padT = pads._1,
      padW = pads._3,
      padH = pads._2,
      withBias = bias,
      wRegularizer = wRegularizer,
      bRegularizer = bRegularizer)
    layer.setInitMethod(weightInitMethod = init, biasInitMethod = Zeros)
    KerasLayer.fuse(layer, activation,
      inputShape).asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object Convolution3D {
  def apply[@specialized(Float, Double) T: ClassTag](
    nbFilter: Int,
    kernelDim1: Int,
    kernelDim2: Int,
    kernelDim3: Int,
    init: String = "glorot_uniform",
    activation: String = null,
    borderMode: String = "valid",
    subsample: (Int, Int, Int) = (1, 1, 1),
    dimOrdering: String = "th",
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    bias: Boolean = true,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): Convolution3D[T] = {
    new Convolution3D[T](nbFilter, kernelDim1, kernelDim2, kernelDim3,
      KerasUtils.getInitMethod(init), KerasUtils.getKerasActivation(activation),
      borderMode, Array(subsample._1, subsample._2, subsample._3),
      KerasUtils.toBigDLFormat5D(dimOrdering),
      wRegularizer, bRegularizer, bias, inputShape)
  }
}
