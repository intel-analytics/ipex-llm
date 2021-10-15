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

import com.intel.analytics.bigdl.dllib.nn.keras.{KerasLayer, Convolution3D => BigDLConvolution3D}
import com.intel.analytics.bigdl.dllib.nn.{InitializationMethod, Xavier}
import com.intel.analytics.bigdl.dllib.optim.Regularizer
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.Net
import com.intel.analytics.bigdl.dllib.keras.layers.utils.KerasUtils

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
 * @param inputShape A Single Shape, does not include the batch dimension.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class Convolution3D[T: ClassTag](
    override val nbFilter: Int,
    override val kernelDim1: Int,
    override val kernelDim2: Int,
    override val kernelDim3: Int,
    override val init: InitializationMethod = Xavier,
    override val activation: KerasLayer[Tensor[T], Tensor[T], T] = null,
    override val borderMode: String = "valid",
    override val subsample: Array[Int] = Array(1, 1, 1),
    override val dimOrdering: String = "CHANNEL_FIRST",
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    override val bias: Boolean = true,
    override val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends BigDLConvolution3D[T](
    nbFilter, kernelDim1, kernelDim2, kernelDim3, init, activation, borderMode,
    subsample, dimOrdering, wRegularizer, bRegularizer, bias, inputShape) with Net {
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
    val subsampleArray = subsample match {
      case null => throw new IllegalArgumentException("For Convolution3D, " +
        "subsample can not be null, please input int tuple of length 3")
      case _ => Array(subsample._1, subsample._2, subsample._3)
    }
    new Convolution3D[T](nbFilter, kernelDim1, kernelDim2, kernelDim3,
      KerasUtils.getInitMethod(init), KerasUtils.getKerasActivation(activation),
      borderMode, Array(subsample._1, subsample._2, subsample._3),
      KerasUtils.toBigDLFormat5D(dimOrdering),
      wRegularizer, bRegularizer, bias, inputShape)
  }
}
