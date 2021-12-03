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

import com.intel.analytics.bigdl.dllib.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.dllib.nn.internal.KerasLayer
import com.intel.analytics.bigdl.dllib.nn.LookupTableSparse
import com.intel.analytics.bigdl.dllib.optim.Regularizer
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.nn.{AddConstant, InitializationMethod, LookupTable, RandomUniform, Zeros, Sequential => TSequential}
import com.intel.analytics.bigdl.dllib.keras.Net
import com.intel.analytics.bigdl.dllib.keras.layers.utils.KerasUtils

import scala.reflect.ClassTag

/**
 * SparseEmbedding is the sparse version of layer Embedding.
 *
 * The input of SparseEmbedding should be a 2D SparseTensor or two 2D sparseTensors.
 * If the input is a SparseTensor, the values are positive integer ids,
 * values in each row of this SparseTensor will be turned into a dense vector.
 * If the input is two SparseTensors, the first tensor should be the integer ids, just
 * like the SparseTensor input. And the second tensor is the corresponding
 * weights of the integer ids.
 *
 * This layer can only be used as the first layer in a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param inputDim Int > 0. Size of the vocabulary.
 * @param outputDim Int >= 0. Dimension of the dense embedding.
 * @param init Initialization method for the weights of the layer. Default is RandomUniform.
 *             You can also pass in corresponding string representations such as 'uniform'
 *             or 'normal', etc. for simple init methods in the factory method.
 * @param combiner A string specifying the reduce type.
 *                 Currently "mean", "sum", "sqrtn" is supported.
 * @param maxNorm If provided, each embedding is normalized to have l2 norm equal to
 *                maxNorm before combining.
 * @param wRegularizer An instance of [[Regularizer]], (eg. L1 or L2 regularization),
 *                     applied to the embedding matrix. Default is null.
 * @param inputShape A Single Shape, does not include the batch dimension.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class SparseEmbedding[T: ClassTag](
    val inputDim: Int,
    val outputDim: Int,
    val combiner: String = "sum",
    val maxNorm: Double = -1,
    val init: InitializationMethod = RandomUniform,
    var wRegularizer: Regularizer[T] = null,
    val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasUtils.addBatch(inputShape)) with Net {

  override def computeOutputShape(inputShape: Shape): Shape = {
    val input = inputShape.toSingle().toArray
    require(input.length == 2,
      s"Embedding requires 2D input, but got input dim ${input.length}")
    Shape(input(0), input(1), outputDim)
  }

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val input = inputShape.toSingle().toArray
    val model = TSequential[T]()
    val layer = LookupTableSparse(
      nIndex = inputDim,
      nOutput = outputDim,
      combiner = combiner,
      maxNorm = maxNorm,
      wRegularizer = wRegularizer)
    layer.setInitMethod(weightInitMethod = init)
    model.add(layer)
    model.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object SparseEmbedding {
  def apply[@specialized(Float, Double) T: ClassTag](
      inputDim: Int,
      outputDim: Int,
      combiner: String = "sum",
      maxNorm: Double = -1,
      init: String = "uniform",
      wRegularizer: Regularizer[T] = null,
      inputShape: Shape = null)(implicit ev: TensorNumeric[T]): SparseEmbedding[T] = {
    new SparseEmbedding[T](inputDim, outputDim, combiner.toLowerCase,
      maxNorm, KerasUtils.getInitMethod(init), wRegularizer, inputShape)
  }
}

