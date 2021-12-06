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

import com.intel.analytics.bigdl.dllib.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.dllib.nn.internal.{KerasLayer, AtrousConvolution2D => BigDLAtrousConvolution2D}
import com.intel.analytics.bigdl.dllib.nn.{InitializationMethod, Xavier}
import com.intel.analytics.bigdl.dllib.optim.Regularizer
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.Net
import com.intel.analytics.bigdl.dllib.keras.layers.utils.KerasUtils

import scala.reflect.ClassTag

/**
 * Applies an atrous convolution operator for filtering windows of 2-D inputs.
 * A.k.a dilated convolution or convolution with holes.
 * Bias will be included in this layer.
 * Data format currently supported for this layer is DataFormat.NCHW (dimOrdering='th').
 * Border mode currently supported for this layer is 'valid'.
 * You can also use AtrousConv2D as an alias of this layer.
 * The input of this layer should be 4D.
 *
 * When using this layer as the first layer in a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 * e.g. input_shape=Shape(3, 128, 128) for 128x128 RGB pictures.
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
 * @param subsample Int array of length 2. Factor by which to subsample output.
 *                  Also called strides elsewhere. Default is (1, 1).
 * @param atrousRate Int array of length 2. Factor for kernel dilation.
 *                   Also called filter_dilation elsewhere. Default is (1, 1).
 * @param dimOrdering Format of input data. Please use DataFormat.NCHW (dimOrdering='th').
 * @param wRegularizer An instance of [[Regularizer]], (eg. L1 or L2 regularization),
 *                     applied to the input weights matrices. Default is null.
 * @param bRegularizer An instance of [[Regularizer]], applied to the bias. Default is null.
 * @param inputShape A Single Shape, does not include the batch dimension.
 * @tparam T The numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class AtrousConvolution2D[T: ClassTag](
   override val nbFilter: Int,
   override val nbRow: Int,
   override val nbCol: Int,
   override val init: InitializationMethod = Xavier,
   override val activation: KerasLayer[Tensor[T], Tensor[T], T] = null,
   override val subsample: Array[Int] = Array(1, 1),
   override val atrousRate: Array[Int] = Array(1, 1),
   override val dimOrdering: DataFormat = DataFormat.NCHW,
   wRegularizer: Regularizer[T] = null,
   bRegularizer: Regularizer[T] = null,
   override val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends BigDLAtrousConvolution2D[T](nbFilter, nbRow, nbCol, init, activation,
                                      subsample, atrousRate, dimOrdering,
                                      wRegularizer, bRegularizer, inputShape) with Net {}

object AtrousConvolution2D {
  def apply[@specialized(Float, Double) T: ClassTag](
    nbFilter: Int,
    nbRow: Int,
    nbCol: Int,
    init: String = "glorot_uniform",
    activation: String = null,
    subsample: (Int, Int) = (1, 1),
    atrousRate: (Int, Int) = (1, 1),
    dimOrdering: String = "th",
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): AtrousConvolution2D[T] = {
    val subsampleArray = subsample match {
      case null => throw new IllegalArgumentException("For AtrousConvolution2D, " +
        "subsample can not be null, please input int tuple of length 2")
      case _ => Array(subsample._1, subsample._2)
    }
    val atrousRateArray = atrousRate match {
      case null => throw new IllegalArgumentException("For AtrousConvolution2D, " +
        "atrousRate can not be null, please input int tuple of length 2")
      case _ => Array(atrousRate._1, atrousRate._2)
    }
    new AtrousConvolution2D[T](nbFilter, nbRow, nbCol, KerasUtils.getInitMethod(init),
      KerasUtils.getKerasActivation(activation),
      subsampleArray, atrousRateArray,
      KerasUtils.toBigDLFormat(dimOrdering), wRegularizer, bRegularizer, inputShape)
  }
}
