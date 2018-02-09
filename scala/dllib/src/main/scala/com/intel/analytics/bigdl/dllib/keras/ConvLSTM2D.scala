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

import com.intel.analytics.bigdl.nn.{ConvLSTMPeephole, Reverse, Select, Sequential => TSequential}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, TensorModule}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape

import scala.reflect.ClassTag

/**
 * Convolutional LSTM.
 * Data format currently supported for this layer is 'CHANNEL_FIRST' (dimOrdering='th').
 * BorderMode of this layer will be 'same'.
 * The convolution kernel for this layer is a square kernel.
 * The input of this layer should be 5D.
 *
 * When using this layer as the first layer in a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param nbFilter Number of convolution filters to use.
 * @param nbKernel Size of the convolution kernel. Integer.
 * @param activation Activation function to use.
 *                   You can also pass in corresponding string representations such as 'relu'
 *                   or 'sigmoid', etc. for simple activations in the factory method.
 *                   Default is 'tanh'.
 * @param innerActivation Activation function for inner cells.
 *                        You can also pass in corresponding string representations such as 'relu'
 *                        or 'sigmoid', etc. for simple activations in the factory method.
 *                        Default is 'hard_sigmoid'.
 * @param dimOrdering Format of input data. Please use "CHANNEL_FIRST" (dimOrdering='th').
 * @param subsample Int. Default is 1. Factor by which to subsample output.
 *                  Also called strides elsewhere.
 * @param wRegularizer An instance of [[Regularizer]], (eg. L1 or L2 regularization),
 *                     applied to the input weights matrices. Default is null.
 * @param uRegularizer An instance of [[Regularizer]], (eg. L1 or L2 regularization),
 *                     applied to the recurrent weights matrices. Default is null.
 * @param bRegularizer An instance of [[Regularizer]], applied to the bias. Default is null.
 * @param returnSequences Boolean. Default is False. Whether to return the last
 *                        output in the output sequence, or the full sequence.
 * @param goBackwards Boolean. Default is False. If True, process the input sequence backwards.
 * @tparam T The numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class ConvLSTM2D[T: ClassTag](
   val nbFilter: Int,
   val nbKernel: Int,
   val activation: AbstractModule[Tensor[T], Tensor[T], T] = null,
   val innerActivation: AbstractModule[Tensor[T], Tensor[T], T] = null,
   val dimOrdering: String = "CHANNEL_FIRST",
   val subsample: Int = 1,
   var wRegularizer: Regularizer[T] = null,
   var uRegularizer: Regularizer[T] = null,
   var bRegularizer: Regularizer[T] = null,
   val returnSequences: Boolean = false,
   val goBackwards: Boolean = false,
   val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasLayer.addBatch(inputShape)) {

  require(dimOrdering.toLowerCase() == "channel_first", s"ConvLSTM2D currently only supports " +
    s"format CHANNEL_FIRST, but got format $dimOrdering")

  override def computeOutputShape(inputShape: Shape): Shape = {
    val input = inputShape.toSingle().toArray
    require(input.length == 5,
      s"ConvLSTM2D requires 5D input, but got input dim ${input.length}")
    val rows = KerasUtils.computeConvOutputLength(input(3), nbKernel, "same", subsample)
    val cols = KerasUtils.computeConvOutputLength(input(4), nbKernel, "same", subsample)
    if (returnSequences) Shape(input(0), input(1), nbFilter, rows, cols)
    else Shape(input(0), nbFilter, rows, cols)
  }

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val input = inputShape.toSingle().toArray
    val model = TSequential[T]()
    if (goBackwards) model.add(Reverse(2))
    val rec = com.intel.analytics.bigdl.nn.Recurrent[T]()
    val layer = ConvLSTMPeephole(
      inputSize = input(2),
      outputSize = nbFilter,
      kernelI = nbKernel,
      kernelC = nbKernel,
      stride = subsample,
      activation = activation.asInstanceOf[TensorModule[T]],
      innerActivation = innerActivation.asInstanceOf[TensorModule[T]],
      wRegularizer = wRegularizer,
      uRegularizer = uRegularizer,
      bRegularizer = bRegularizer,
      withPeephole = false)
    rec.add(layer)
    model.add(rec)
    if (!returnSequences) model.add(Select(2, -1))
    model.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object ConvLSTM2D {
  def apply[@specialized(Float, Double) T: ClassTag](
    nbFilter: Int,
    nbKernel: Int,
    activation: String = "tanh",
    innerActivation: String = "hard_sigmoid",
    dimOrdering: String = "th",
    subsample: Int = 1,
    wRegularizer: Regularizer[T] = null,
    uRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    returnSequences: Boolean = false,
    goBackwards: Boolean = false,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): ConvLSTM2D[T] = {
    new ConvLSTM2D[T](nbFilter, nbKernel, KerasUtils.getActivation(activation),
      KerasUtils.getActivation(innerActivation),
      KerasUtils.toBigDLFormat5D(dimOrdering),
      subsample, wRegularizer, uRegularizer, bRegularizer,
      returnSequences, goBackwards, inputShape)
  }
}
