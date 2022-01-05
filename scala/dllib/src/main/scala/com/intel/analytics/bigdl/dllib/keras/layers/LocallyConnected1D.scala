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

import com.intel.analytics.bigdl.dllib.nn.internal.{KerasLayer, LocallyConnected1D => BigDLLocallyConnected1D}
import com.intel.analytics.bigdl.dllib.optim.Regularizer
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.Net
import com.intel.analytics.bigdl.dllib.keras.layers.utils.KerasUtils

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
 * @param inputShape A Single Shape, does not include the batch dimension.
 * @tparam T The numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class LocallyConnected1D[T: ClassTag](
   override val nbFilter: Int,
   override val filterLength: Int,
   override val activation: KerasLayer[Tensor[T], Tensor[T], T] = null,
   override val subsampleLength: Int = 1,
   wRegularizer: Regularizer[T] = null,
   bRegularizer: Regularizer[T] = null,
   override val bias: Boolean = true,
   override val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends BigDLLocallyConnected1D[T](nbFilter, filterLength, activation, subsampleLength,
                                      wRegularizer, bRegularizer, bias, inputShape) with Net {}

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
