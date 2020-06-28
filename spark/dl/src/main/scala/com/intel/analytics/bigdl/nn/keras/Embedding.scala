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

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.nn.{AddConstant, InitializationMethod, LookupTable, RandomUniform, Zeros, Sequential => TSequential}

import scala.reflect.ClassTag

/**
 * Turn positive integers (indexes) into dense vectors of fixed size.
 * The input of this layer should be 2D.
 *
 * This layer can only be used as the first layer in a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param inputDim Int > 0. Size of the vocabulary.
 * @param outputDim Int >= 0. Dimension of the dense embedding.
 * @param init Initialization method for the weights of the layer. Default is RandomUniform.
 *             You can also pass in corresponding string representations such as 'uniform'
 *             or 'normal', etc. for simple init methods in the factory method.
 * @param wRegularizer An instance of [[Regularizer]], (eg. L1 or L2 regularization),
 *                     applied to the embedding matrix. Default is null.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class Embedding[T: ClassTag](
   val inputDim: Int,
   val outputDim: Int,
   val init: InitializationMethod = RandomUniform,
   var wRegularizer: Regularizer[T] = null,
   val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasLayer.addBatch(inputShape)) {

  override def computeOutputShape(inputShape: Shape): Shape = {
    val input = inputShape.toSingle().toArray
    require(input.length == 2,
      s"Embedding requires 2D input, but got input dim ${input.length}")
    Shape(input(0), input(1), outputDim)
  }

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val input = inputShape.toSingle().toArray
    val model = TSequential[T]()
    model.add(AddConstant(1.0))
    val layer = LookupTable(
      nIndex = inputDim,
      nOutput = outputDim,
      wRegularizer = wRegularizer)
    layer.setInitMethod(weightInitMethod = init)
    model.add(layer)
    model.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object Embedding {
  def apply[@specialized(Float, Double) T: ClassTag](
    inputDim: Int,
    outputDim: Int,
    init: String = "uniform",
    wRegularizer: Regularizer[T] = null,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): Embedding[T] = {
    new Embedding[T](inputDim, outputDim, KerasUtils.getInitMethod(init),
      wRegularizer, inputShape)
  }
}
