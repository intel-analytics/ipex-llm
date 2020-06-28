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

import com.intel.analytics.bigdl.nn.{InitializationMethod, SpatialSeparableConvolution, Xavier}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, DataFormat}

import scala.reflect.ClassTag

/**
 * Applies separable convolution operator for 2D inputs.
 * Separable convolutions consist in first performing a depthwise spatial convolution (which acts
 * on each input channel separately) followed by a pointwise convolution which mixes together the
 * resulting output channels. The depthMultiplier argument controls how many output channels are
 * generated per input channel in the depthwise step.
 * You can also use SeparableConv2D as an alias of this layer.
 * The input of this layer should be 4D.
 *
 * When using this layer as the first layer in a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 * e.g. inputShape=Shape(3, 128, 128) for 128x128 RGB pictures.
 *
 * @param nbFilter Number of convolution filters to use.
 * @param nbRow Number of rows in the convolution kernel.
 * @param nbCol Number of columns in the convolution kernel.
 * @param init Initialization method for the weights of the layer. Default is Xavier.
 *             You can also pass in corresponding string representations such as 'glorot_uniform'
 *             or 'normal', etc. for simple init methods in the factory method.
 * @param activation Activation function to use. Default is null.
 *                   You can also pass in corresponding string representations such as 'relu'
 *                   or 'sigmoid', etc. for simple activations in the factory method.
 * @param borderMode Either 'valid' or 'same'. Default is 'valid'.
 * @param subsample Int array of length 2 corresponding to the step of the convolution in the height
 *                  and width dimension. Also called strides elsewhere. Default is (1, 1).
 * @param depthMultiplier How many output channel to use per input channel
 *                        for the depthwise convolution step. Integer. Default is 1.
 * @param dimOrdering Format of input data. Either DataFormat.NCHW (dimOrdering='th') or
 *                    DataFormat.NHWC (dimOrdering='tf'). Default is NCHW.
 * @param depthwiseRegularizer An instance of [[Regularizer]], (eg. L1 or L2 regularization),
 *                             applied to the depthwise weights matrices. Default is null.
 * @param pointwiseRegularizer An instance of [[Regularizer]], applied to the pointwise weights
 *                             matrices. Default is null.
 * @param bRegularizer An instance of [[Regularizer]], applied to the bias. Default is null.
 * @param bias Whether to include a bias (i.e. make the layer affine rather than linear).
 *             Default is true.
 * @tparam T The numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class SeparableConvolution2D[T: ClassTag](
   val nbFilter: Int,
   val nbRow: Int,
   val nbCol: Int,
   val init: InitializationMethod = Xavier,
   val activation: KerasLayer[Tensor[T], Tensor[T], T] = null,
   val borderMode: String = "valid",
   val subsample: Array[Int] = Array(1, 1),
   val depthMultiplier: Int = 1,
   val dimOrdering: DataFormat = DataFormat.NCHW,
   var depthwiseRegularizer: Regularizer[T] = null,
   var pointwiseRegularizer: Regularizer[T] = null,
   var bRegularizer: Regularizer[T] = null,
   val bias: Boolean = true,
   val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasLayer.addBatch(inputShape)) {

  require(borderMode == "valid" || borderMode == "same", s"Invalid border mode for " +
    s"SeparableConvolution2D: $borderMode")
  require(subsample.length == 2, s"For SeparableConvolution2D, " +
    s"subsample should be of length 2 but got length ${subsample.length}")

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val input = inputShape.toSingle().toArray
    val stackSize = if (dimOrdering == DataFormat.NCHW) input(1) else input(3)
    val pad = KerasUtils.getPadsFromBorderMode(borderMode)
    val layer = SpatialSeparableConvolution(
      nInputChannel = stackSize,
      nOutputChannel = nbFilter,
      depthMultiplier = depthMultiplier,
      kW = nbCol,
      kH = nbRow,
      sW = subsample(1),
      sH = subsample(0),
      pW = pad._2,
      pH = pad._1,
      hasBias = bias,
      dataFormat = dimOrdering,
      wRegularizer = depthwiseRegularizer,
      bRegularizer = bRegularizer,
      pRegularizer = pointwiseRegularizer)
    KerasLayer.fuse(layer, activation,
      inputShape).asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object SeparableConvolution2D {
  def apply[@specialized(Float, Double) T: ClassTag](
    nbFilter: Int,
    nbRow: Int,
    nbCol: Int,
    init: String = "glorot_uniform",
    activation: String = null,
    borderMode: String = "valid",
    subsample: (Int, Int) = (1, 1),
    depthMultiplier: Int = 1,
    dimOrdering: String = "th",
    depthwiseRegularizer: Regularizer[T] = null,
    pointwiseRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    bias: Boolean = true,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]) : SeparableConvolution2D[T] = {
    new SeparableConvolution2D[T](nbFilter, nbRow, nbCol,
      KerasUtils.getInitMethod(init), KerasUtils.getKerasActivation(activation),
      borderMode, Array(subsample._1, subsample._2), depthMultiplier,
      KerasUtils.toBigDLFormat(dimOrdering), depthwiseRegularizer,
      pointwiseRegularizer, bRegularizer, bias, inputShape)
  }
}
