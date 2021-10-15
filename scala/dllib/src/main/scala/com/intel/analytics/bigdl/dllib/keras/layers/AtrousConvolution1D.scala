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

import com.intel.analytics.bigdl.dllib.nn.keras.{KerasLayer, AtrousConvolution1D => BigDLAtrousConvolution1D}
import com.intel.analytics.bigdl.dllib.nn.{InitializationMethod, Xavier}
import com.intel.analytics.bigdl.dllib.optim.Regularizer
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.Net
import com.intel.analytics.bigdl.dllib.keras.layers.utils.KerasUtils

import scala.reflect.ClassTag

/**
 * Applies an atrous convolution operator for filtering neighborhoods of 1-D inputs.
 * A.k.a dilated convolution or convolution with holes.
 * Bias will be included in this layer.
 * Border mode currently supported for this layer is 'valid'.
 * You can also use AtrousConv1D as an alias of this layer.
 * The input of this layer should be 3D.
 *
 * When using this layer as the first layer in a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param nbFilter Number of convolution kernels to use.
 * @param filterLength The extension (spatial or temporal) of each filter.
 * @param init Initialization method for the weights of the layer. Default is Xavier.
 *             You can also pass in corresponding string representations such as 'glorot_uniform'
 *             or 'normal', etc. for simple init methods in the factory method.
 * @param activation Activation function to use. Default is null.
 *                   You can also pass in corresponding string representations such as 'relu'
 *                   or 'sigmoid', etc. for simple activations in the factory method.
 * @param subsampleLength Factor by which to subsample output. Integer. Default is 1.
 * @param atrousRate Factor for kernel dilation. Also called filter_dilation elsewhere.
 *                   Integer. Default is 1.
 * @param wRegularizer An instance of [[Regularizer]], (eg. L1 or L2 regularization),
 *                     applied to the input weights matrices. Default is null.
 * @param bRegularizer An instance of [[Regularizer]], applied to the bias. Default is null.
 * @param inputShape A Single Shape, does not include the batch dimension.
 * @tparam T The numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class AtrousConvolution1D[T: ClassTag](
   override val nbFilter: Int,
   override val filterLength: Int,
   override val init: InitializationMethod = Xavier,
   override val activation: KerasLayer[Tensor[T], Tensor[T], T] = null,
   override val subsampleLength: Int = 1,
   override val atrousRate: Int = 1,
   wRegularizer: Regularizer[T] = null,
   bRegularizer: Regularizer[T] = null,
   override val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends BigDLAtrousConvolution1D[T](nbFilter, filterLength, init, activation, subsampleLength,
    atrousRate, wRegularizer, bRegularizer, inputShape) with Net {}

object AtrousConvolution1D {
  def apply[@specialized(Float, Double) T: ClassTag](
    nbFilter: Int,
    filterLength: Int,
    init: String = "glorot_uniform",
    activation: String = null,
    subsampleLength: Int = 1,
    atrousRate: Int = 1,
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): AtrousConvolution1D[T] = {
    new AtrousConvolution1D[T](nbFilter, filterLength, KerasUtils.getInitMethod(init),
      KerasUtils.getKerasActivation(activation), subsampleLength, atrousRate,
      wRegularizer, bRegularizer, inputShape)
  }
}
