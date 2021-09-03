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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, DataFormat}
import com.intel.analytics.bigdl.nn.{InitializationMethod, SpatialFullConvolution, Xavier, Zeros}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape

import scala.reflect.ClassTag

/**
 * Transposed convolution operator for filtering windows of 2-D inputs.
 * The need for transposed convolutions generally arises from the desire to use a transformation
 * going in the opposite direction of a normal convolution, i.e., from something that has
 * the shape of the output of some convolution to something that has the shape of its input
 * while maintaining a connectivity pattern that is compatible with said convolution.
 * Data format currently supported for this layer is DataFormat.NCHW (dimOrdering='th').
 * Border mode currently supported for this layer is 'valid'.
 * You can also use Deconv2D as an alias of this layer.
 * The input of this layer should be 4D.
 *
 * When using this layer as the first layer in a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 * e.g. inputShape=Shape(3, 128, 128) for 128x128 RGB pictures.
 *
 * @param nbFilter Number of transposed convolution filters to use.
 * @param nbRow Number of rows in the transposed convolution kernel.
 * @param nbCol Number of columns in the transposed convolution kernel.
 * @param init Initialization method for the weights of the layer. Default is Xavier.
 *             You can also pass in corresponding string representations such as 'glorot_uniform'
 *             or 'normal', etc. for simple init methods in the factory method.
 * @param activation Activation function to use. Default is null.
 *                   You can also pass in corresponding string representations such as 'relu'
 *                   or 'sigmoid', etc. for simple activations in the factory method.
 * @param subsample Int array of length 2. The step of the convolution in the height and
 *                  width dimension. Also called strides elsewhere. Default is (1, 1).
 * @param dimOrdering Format of input data. Please use DataFormat.NCHW (dimOrdering='th').
 * @param wRegularizer An instance of [[Regularizer]], (eg. L1 or L2 regularization),
 *                     applied to the input weights matrices. Default is null.
 * @param bRegularizer An instance of [[Regularizer]], applied to the bias. Default is null.
 * @param bias Whether to include a bias (i.e. make the layer affine rather than linear).
 *             Default is true.
 * @tparam T The numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class Deconvolution2D[T: ClassTag](
   val nbFilter: Int,
   val nbRow: Int,
   val nbCol: Int,
   val init: InitializationMethod = Xavier,
   val activation: KerasLayer[Tensor[T], Tensor[T], T] = null,
   val subsample: Array[Int] = Array(1, 1),
   val dimOrdering: DataFormat = DataFormat.NCHW,
   var wRegularizer: Regularizer[T] = null,
   var bRegularizer: Regularizer[T] = null,
   val bias: Boolean = true,
   val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasLayer.addBatch(inputShape)) {

  require(dimOrdering == DataFormat.NCHW, s"Deconvolution2D currently only supports " +
    s"format NCHW, but got format $dimOrdering")
  require(subsample.length == 2,
    s"For Deconvolution2D, subsample should be of length 2 but got length ${subsample.length}")

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val input = inputShape.toSingle().toArray
    val layer = SpatialFullConvolution(
      nInputPlane = input(1),
      nOutputPlane = nbFilter,
      kW = nbCol,
      kH = nbRow,
      dW = subsample(1),
      dH = subsample(0),
      noBias = !bias,
      wRegularizer = wRegularizer,
      bRegularizer = bRegularizer)
    layer.setInitMethod(weightInitMethod = init, biasInitMethod = Zeros)
    KerasLayer.fuse(layer, activation,
      inputShape).asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object Deconvolution2D {
  def apply[@specialized(Float, Double) T: ClassTag](
    nbFilter: Int,
    nbRow: Int,
    nbCol: Int,
    init: String = "glorot_uniform",
    activation: String = null,
    subsample: (Int, Int) = (1, 1),
    dimOrdering: String = "th",
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    bias: Boolean = true,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): Deconvolution2D[T] = {
    new Deconvolution2D[T](nbFilter, nbRow, nbCol, KerasUtils.getInitMethod(init),
      KerasUtils.getKerasActivation(activation), Array(subsample._1, subsample._2),
      KerasUtils.toBigDLFormat(dimOrdering), wRegularizer,
      bRegularizer, bias, inputShape)
  }
}
