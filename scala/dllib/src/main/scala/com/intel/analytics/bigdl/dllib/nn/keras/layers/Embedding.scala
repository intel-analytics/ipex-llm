/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.pipeline.api.keras.layers

import com.intel.analytics.bigdl.nn.keras.{Embedding => BEmbedding}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.nn.{InitializationMethod, RandomUniform}
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils

import scala.reflect.ClassTag

/**
 * Turn non-negative integers (indices) into dense vectors of fixed size.
 * The input of this layer should be 2D.
 *
 * This layer can only be used as the first layer in a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param inputDim Int > 0. Size of the vocabulary, ie. 1 + maximum integer
 *                 index occurring in the input data.
 *                 Each word index in the input should be within range [0, inputDim-1].
 * @param outputDim Int >= 0. Dimension of the dense embedding.
 * @param init Initialization method for the weights of the layer. Default is RandomUniform.
 *             You can also pass in corresponding string representations such as 'uniform'
 *             or 'normal', etc. for simple init methods in the factory method.
 * @param wRegularizer An instance of [[Regularizer]], (eg. L1 or L2 regularization),
 *                     applied to the embedding matrix. Default is null.
 * @param inputShape A Single Shape, does not include the batch dimension.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class Embedding[T: ClassTag](
    override val inputDim: Int,
    override val outputDim: Int,
    override val init: InitializationMethod = RandomUniform,
    wRegularizer: Regularizer[T] = null,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends BEmbedding[T] (
    inputDim, outputDim, init, wRegularizer, inputShape) with Net {
}

object Embedding {
  def apply[@specialized(Float, Double) T: ClassTag](
      inputDim: Int,
      outputDim: Int,
      init: String = "uniform",
      wRegularizer: Regularizer[T] = null,
      inputLength: Int)(implicit ev: TensorNumeric[T]): Embedding[T] = {
    new Embedding[T](inputDim, outputDim, KerasUtils.getInitMethod(init),
      wRegularizer, Shape(inputLength))
  }
}
