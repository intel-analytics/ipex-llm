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

import com.intel.analytics.bigdl.nn.{Cell, RnnCell}
import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.nn.keras.KerasLayer
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils

import scala.reflect.ClassTag

/**
 * Long Short Term Memory unit architecture.
 * The input of this layer should be 3D, i.e. (batch, time steps, input dim).
 *
 * When you use this layer as the first layer of a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param outputDimension Hidden unit size. Dimension of internal projections and final output.
 * @param activation Activation function to use.
 *                   You can also pass in corresponding string representations such as 'relu'
 *                   or 'sigmoid', etc. for simple activations in the factory method.
 *                   Default is 'tanh'.
 * @param innerActivation Activation function for inner cells.
 *                        You can also pass in corresponding string representations such as 'relu'
 *                        or 'sigmoid', etc. for simple activations in the factory method.
 *                        Default is 'hard_sigmoid'.
 * @param returnSeq Whether to return the full sequence or only return the last output
 *                        in the output sequence. Default is false.
 * @param goBackward Whether the input sequence will be processed backwards. Default is false.
 * @param wRegularizer An instance of [[Regularizer]], (eg. L1 or L2 regularization),
 *                     applied to the input weights matrices. Default is null.
 * @param uRegularizer An instance of [[Regularizer]], applied the recurrent weights matrices.
 *                     Default is null.
 * @param bRegularizer An instance of [[Regularizer]], applied to the bias. Default is null.
 * @param mInputShape A Single Shape, does not include the batch dimension.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class LSTM[T: ClassTag](
   var outputDimension: Int,
   val activation: KerasLayer[Tensor[T], Tensor[T], T] = null,
   val innerActivation: KerasLayer[Tensor[T], Tensor[T], T] = null,
   var returnSeq: Boolean = false,
   var goBackward: Boolean = false,
   var wRegularizer: Regularizer[T] = null,
   var uRegularizer: Regularizer[T] = null,
   var bRegularizer: Regularizer[T] = null,
   var mInputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends Recurrent[T] (
    outputDimension, returnSeq, goBackward, mInputShape) with Net {

  override def buildCell(input: Array[Int]): Cell[T] = {
    com.intel.analytics.bigdl.nn.LSTM[T](
      inputSize = input(2),
      hiddenSize = outputDim,
      activation = activation.doBuild(inputShape).asInstanceOf[TensorModule[T]],
      innerActivation = innerActivation.doBuild(inputShape).asInstanceOf[TensorModule[T]],
      wRegularizer = wRegularizer,
      uRegularizer = uRegularizer,
      bRegularizer = bRegularizer)
  }

  override private[zoo] def toKeras2(): String = {
    val params = Net.inputShapeToString(inputShape) ++
      Net.activationToString(activation) ++
      Net.activationToString(innerActivation, "recurrent_activation") ++
      Net.param(returnSeq, "return_sequences") ++
      Net.param(outputDimension, "units")
      Net.param(getName())
    Net.kerasDef(this, params)
  }

  override private[zoo] def getKerasWeights(): Array[Tensor[Float]] = {
    val weights = this.parameters()._1
    val kWeights = Array.tabulate(weights.length)(_ => Tensor[Float]())
    weights(0) = weights(0).t().contiguous()
    weights(2) = weights(2).t().contiguous()
    weights(0).cast[Float](kWeights(0).resizeAs(weights(0)))
    weights(2).cast[Float](kWeights(1).resizeAs(weights(2)))
    weights(1).cast[Float](kWeights(2).resizeAs(weights(1)))
    // map to keras's weight
    switch(kWeights(0), 2)
    switch(kWeights(1), 2)
    switch(kWeights(2), 1)

    kWeights
  }

  private def switch(t: Tensor[Float], dim: Int): Unit = {
    val tmpWeight = t.narrow(dim, 1, outputDimension).clone()
    tmpWeight.copy(t.narrow(dim, 1 + outputDimension, outputDimension))
    t.narrow(dim, 1 + outputDimension, outputDimension)
      .copy(t.narrow(dim, 2 * outputDimension + 1, outputDimension))
    t.narrow(dim, 2 * outputDimension + 1, outputDimension).copy(tmpWeight)
  }

}

object LSTM {
  def apply[@specialized(Float, Double) T: ClassTag](
    outputDim: Int,
    activation: String = "tanh",
    innerActivation: String = "hard_sigmoid",
    returnSequences: Boolean = false,
    goBackwards: Boolean = false,
    wRegularizer: Regularizer[T] = null,
    uRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]) : LSTM[T] = {
    new LSTM(outputDim, KerasUtils.getKerasActivation(activation),
      KerasUtils.getKerasActivation(innerActivation), returnSequences,
      goBackwards, wRegularizer, uRegularizer, bRegularizer, inputShape)
  }
}
