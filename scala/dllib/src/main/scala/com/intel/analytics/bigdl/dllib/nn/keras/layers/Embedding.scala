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

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.nn.keras.{Embedding => BEmbedding}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.nn.{AddConstant => TAddConstant, InitializationMethod, LookupTable, RandomUniform, Sequential => TSequential}
import com.intel.analytics.bigdl.tensor.Tensor
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
 * @param outputDim Int > 0. Dimension of the dense embedding.
 * @param init Initialization method for the weights of the layer. Default is RandomUniform.
 *             You can also pass in corresponding string representations such as 'uniform'
 *             or 'normal', etc. for simple init methods in the factory method.
 * @param initWeights Tensor. Initial weights set to this layer, which should be a Tensor of
 *                size (inputDim, outputDim). Default is null and in this case weights are
 *                initialized by the initialization method specified by 'init'.
 *                Otherwise, 'weights' will override 'init' to take effect.
 * @param trainable Whether this layer is trainable or not. Default is true.
 * @param wRegularizer An instance of [[Regularizer]], (eg. L1 or L2 regularization),
 *                     applied to the embedding matrix. Default is null.
 * @param inputShape A Single Shape, does not include the batch dimension.
 * @param maskZero: if maskZero is set to true, the input whose value equals `paddingValue`
 *                the output will be masked to zero vector.
 * @param paddingValue padding value, default 0
 * @param zeroBasedId default true and input should be 0 based. Otherwise need to be 1 base
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class Embedding[T: ClassTag](
                              override val inputDim: Int,
                              override val outputDim: Int,
                              override val init: InitializationMethod = RandomUniform,
                              val initWeights: Tensor[T] = null,
                              val trainable: Boolean = true,
                              wRegularizer: Regularizer[T] = null,
                              inputShape: Shape = null,
                              maskZero: Boolean = false,
                              paddingValue: Int = 0,
                              zeroBasedId: Boolean = true
                              )(implicit ev: TensorNumeric[T])
  extends BEmbedding[T] (
    inputDim, outputDim, init, wRegularizer, inputShape) with Net {

  require(inputDim > 0, s"inputDim of Embedding must be a positive integer, but got $inputDim")
  require(outputDim > 0, s"outputDim of Embedding must be a positive integer, but got $outputDim")

  if (initWeights != null) {
    require(initWeights.size().sameElements(Array(inputDim, outputDim)),
    "weights size should match (inputDim, outputDim)")
  }

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val model = TSequential[T]()
    if (zeroBasedId) {
      model.add(TAddConstant(1.0))
    }
    val layer = LookupTable(
      nIndex = inputDim,
      nOutput = outputDim,
      wRegularizer = wRegularizer,
      maskZero = maskZero,
      paddingValue = paddingValue)
    if (initWeights != null) {
      layer.setWeightsBias(Array(initWeights))
    }
    else {
      layer.setInitMethod(weightInitMethod = init, biasInitMethod = init)
    }
    if (! trainable) layer.freeze()
    model.add(layer)
    model.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }

  override private[zoo] def toKeras2(): String = {
    val params = Net.inputShapeToString(inputShape) ++
      Net.param(getName()) ++
      Net.param(inputDim, "input_dim") ++
      Net.param(outputDim, "output_dim") ++
      Net.param(maskZero, "mask_zero")
    Net.kerasDef(this, params)
  }
}

object Embedding {
  def apply[@specialized(Float, Double) T: ClassTag](
    inputDim: Int,
    outputDim: Int,
    init: String = "uniform",
    weights: Tensor[T] = null,
    trainable: Boolean = true,
    wRegularizer: Regularizer[T] = null,
    inputLength: Int = -1,
    maskZero: Boolean = false,
    paddingValue: Int = 0,
    zeroBasedId: Boolean = true
      )(implicit ev: TensorNumeric[T]): Embedding[T] = {
    // Remark: It is possible that inputShape is specified in Input node or layer.
    val shape = if (inputLength > 0) Shape(inputLength) else null
    new Embedding[T](inputDim, outputDim, KerasUtils.getInitMethod(init),
      weights, trainable, wRegularizer, shape, maskZero, paddingValue, zeroBasedId)
  }
}
