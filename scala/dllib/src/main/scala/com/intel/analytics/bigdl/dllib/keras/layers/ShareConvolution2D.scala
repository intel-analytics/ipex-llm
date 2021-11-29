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

import com.intel.analytics.bigdl.dllib.nn.abstractnn.{AbstractModule, DataFormat}
import com.intel.analytics.bigdl.dllib.nn.keras.KerasLayer
import com.intel.analytics.bigdl.dllib.nn.{InitializationMethod, SpatialShareConvolution, Xavier, Zeros}
import com.intel.analytics.bigdl.dllib.optim.Regularizer
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.Net
import com.intel.analytics.bigdl.dllib.keras.layers.utils.KerasUtils

import scala.reflect.ClassTag

/**
 * Applies a 2D convolution over an input image composed of several input planes.
 * You can also use ShareConv2D as an alias of this layer.
 * Data format currently supported for this layer is DataFormat.NCHW (dimOrdering='th').
 * The input of this layer should be 4D.
 *
 * When you use this layer as the first layer of a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension),
 * e.g. inputShape=Shape(3, 128, 128) for 128x128 RGB pictures.
 *
 * Remark: This layer is from Torch and wrapped in Keras style.
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
 * @param subsample Int array of length 2 corresponding to the step of the convolution in the
 *                  height and width dimension. Also called strides elsewhere. Default is (1, 1).
 * @param padH The additional zeros added to the height dimension. Default is 0.
 * @param padW The additional zeros added to the width dimension. Default is 0.
 * @param propagateBack Whether to propagate gradient back. Default is true.
 * @param dimOrdering Format of input data. Please use DataFormat.NCHW (dimOrdering='th').
 * @param wRegularizer An instance of [[Regularizer]], (eg. L1 or L2 regularization),
 *                     applied to the input weights matrices. Default is null.
 * @param bRegularizer An instance of [[Regularizer]], applied to the bias. Default is null.
 * @param bias Whether to include a bias (i.e. make the layer affine rather than linear).
 *             Default is true.
 * @param inputShape A Single Shape, does not include the batch dimension.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class ShareConvolution2D[T: ClassTag](
    val nbFilter: Int,
    val nbRow: Int,
    val nbCol: Int,
    val init: InitializationMethod = Xavier,
    val activation: KerasLayer[Tensor[T], Tensor[T], T] = null,
    val subsample: Array[Int] = Array(1, 1),
    val padH: Int = 0,
    val padW: Int = 0,
    val propagateBack: Boolean = true,
    val dimOrdering: DataFormat = DataFormat.NCHW,
    var wRegularizer: Regularizer[T] = null,
    var bRegularizer: Regularizer[T] = null,
    val bias: Boolean = true,
    val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasUtils.addBatch(inputShape)) with Net {

  require(dimOrdering == DataFormat.NCHW, s"ShareConvolution2D currently only supports " +
    s"format NCHW, but got format $dimOrdering")
  require(subsample.length == 2,
    s"For ShareConvolution2D, subsample should be of length 2 but got length ${subsample.length}")

  override def computeOutputShape(inputShape: Shape): Shape = {
    val input = inputShape.toSingle().toArray
    require(input.length == 4,
      s"ShareConvolution2D requires 4D input, but got input dim ${input.length}")
    val rows = (input(2) + 2 * padH - nbRow) / subsample(0) + 1
    val cols = (input(3) + 2 * padW - nbCol) / subsample(1) + 1
    Shape(input(0), nbFilter, rows, cols)
  }

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val input = inputShape.toSingle().toArray
    val layer = SpatialShareConvolution(
      nInputPlane = input(1),
      nOutputPlane = nbFilter,
      kernelW = nbCol,
      kernelH = nbRow,
      strideW = subsample(1),
      strideH = subsample(0),
      padW = padW,
      padH = padH,
      propagateBack = propagateBack,
      wRegularizer = wRegularizer,
      bRegularizer = bRegularizer,
      withBias = bias)
    layer.setInitMethod(weightInitMethod = init, biasInitMethod = Zeros)
    KerasUtils.fuse(layer, activation,
      inputShape).asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object ShareConvolution2D {
  def apply[@specialized(Float, Double) T: ClassTag](
    nbFilter: Int,
    nbRow: Int,
    nbCol: Int,
    init: String = "glorot_uniform",
    activation: String = null,
    subsample: (Int, Int) = (1, 1),
    padH: Int = 0,
    padW: Int = 0,
    propagateBack: Boolean = true,
    dimOrdering: String = "th",
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    bias: Boolean = true,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): ShareConvolution2D[T] = {
    new ShareConvolution2D[T](nbFilter, nbRow, nbCol,
      KerasUtils.getInitMethod(init), KerasUtils.getKerasActivation(activation),
      Array(subsample._1, subsample._2), padH, padW, propagateBack,
      KerasUtils.toBigDLFormat(dimOrdering), wRegularizer,
      bRegularizer, bias, inputShape)
  }
}
