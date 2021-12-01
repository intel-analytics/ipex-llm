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

import com.intel.analytics.bigdl.dllib.nn.internal.{MaxoutDense => BigDLMaxoutDense}
import com.intel.analytics.bigdl.dllib.optim.Regularizer
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.Net

import scala.reflect.ClassTag

/**
 * A dense maxout layer that takes the element-wise maximum of linear layers.
 * This allows the layer to learn a convex, piecewise linear activation function over the inputs.
 * The input of this layer should be 2D.
 *
 * When you use this layer as the first layer of a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param outputDim The size of output dimension.
 * @param nbFeature Number of Dense layers to use internally. Integer. Default is 4.
 * @param wRegularizer An instance of [[Regularizer]], (eg. L1 or L2 regularization),
 *                     applied to the main weights matrices. Default is null.
 * @param bRegularizer An instance of [[Regularizer]], applied to the bias. Default is null.
 * @param bias Whether to include a bias (i.e. make the layer affine rather than linear).
 *             Default is true.
 * @param inputShape A Single Shape, does not include the batch dimension.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class MaxoutDense[T: ClassTag](
   override val outputDim: Int,
   override val nbFeature: Int = 4,
   override val wRegularizer: Regularizer[T] = null,
   bRegularizer: Regularizer[T] = null,
   override val bias: Boolean = true,
   override val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends BigDLMaxoutDense[T](outputDim, nbFeature, wRegularizer,
    bRegularizer, bias, inputShape) with Net {}

object MaxoutDense {
  def apply[@specialized(Float, Double) T: ClassTag](
    outputDim: Int,
    nbFeature: Int = 4,
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    bias: Boolean = true,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): MaxoutDense[T] = {
    new MaxoutDense[T](outputDim, nbFeature, wRegularizer, bRegularizer, bias, inputShape)
  }
}
