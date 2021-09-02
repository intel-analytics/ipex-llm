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

import com.intel.analytics.bigdl.nn.{Cell}
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
 * Convolutional LSTM.
 * Note that currently only 'same' padding is supported.
 * The convolution kernel for this layer is a square kernel with equal strides.
 * The input of this layer should be 5D, i.e. (samples, time, channels, rows, cols), and
 * 'CHANNEL_FIRST' (dimOrdering='th') is expected.
 *
 * When using this layer as the first layer in a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param outputDimension Number of convolution filters to use.
 * @param nbKernel Number of rows/columns in the convolution kernel. Square kernel.
 * @param activation Activation function to use.
 *                   You can also pass in corresponding string representations such as 'relu'
 *                   or 'sigmoid', etc. for simple activations in the factory method.
 *                   Default is 'tanh'.
 * @param innerActivation Activation function for inner cells.
 *                        You can also pass in corresponding string representations such as 'relu'
 *                        or 'sigmoid', etc. for simple activations in the factory method.
 *                        Default is 'hard_sigmoid'.
 * @param dimOrdering Format of input data. Please use "CHANNEL_FIRST" (dimOrdering='th').
 * @param subsample Factor by which to subsample output.
 *                  Also called strides elsewhere. Default is 1.
 * @param borderMode One of "same" or "valid".
 *                  Also called padding elsewhere. Default is "valid".
 * @param wRegularizer An instance of [[Regularizer]], (eg. L1 or L2 regularization),
 *                     applied to the input weights matrices. Default is null.
 * @param uRegularizer An instance of [[Regularizer]], (eg. L1 or L2 regularization),
 *                     applied to the recurrent weights matrices. Default is null.
 * @param bRegularizer An instance of [[Regularizer]], applied to the bias. Default is null.
 * @param returnSeq Whether to return the full sequence or the last output
 *                  in the output sequence. Default is false.
 * @param goBackward Whether the input sequence will be processed backwards. Default is false.
 * @param mInputShape A Single Shape, does not include the batch dimension.
 * @tparam T The numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class ConvLSTM2D[T: ClassTag](
   var outputDimension: Int,
   val nbKernel: Int,
   val activation: KerasLayer[Tensor[T], Tensor[T], T] = null,
   val innerActivation: KerasLayer[Tensor[T], Tensor[T], T] = null,
   val dimOrdering: String = "CHANNEL_FIRST",
   val subsample: Int = 1,
   val borderMode: String = "valid",
   var wRegularizer: Regularizer[T] = null,
   var uRegularizer: Regularizer[T] = null,
   var bRegularizer: Regularizer[T] = null,
   var returnSeq: Boolean = false,
   var goBackward: Boolean = false,
   var mInputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends Recurrent[T](outputDimension, returnSeq, goBackward, mInputShape) with Net {

  require(dimOrdering.toLowerCase() == "channel_first", s"ConvLSTM2D currently only supports " +
    s"format CHANNEL_FIRST, but got format $dimOrdering")
  require(borderMode.toLowerCase() == "same" || borderMode.toLowerCase() == "valid",
    s"ConvLSTM2D currently only supports " +
    s"same and valid, but got padding $borderMode")

  override def computeOutputShape(inputShape: Shape): Shape = {
    val input = inputShape.toSingle().toArray
    require(input.length == 5,
      s"ConvLSTM2D requires 5D input, but got input dim ${input.length}")
    val rows = KerasUtils.computeConvOutputLength(input(3), nbKernel, borderMode, subsample)
    val cols = KerasUtils.computeConvOutputLength(input(4), nbKernel, borderMode, subsample)
    if (returnSequences) Shape(input(0), input(1), outputDimension, rows, cols)
    else Shape(input(0), outputDimension, rows, cols)
  }

  override def buildCell(input: Array[Int]): Cell[T] = {
    val paddingValue = if (borderMode.toLowerCase == "same") -1
    else 0
    new InternalConvLSTM2D[T](
      inputSize = input(2),
      outputSize = outputDimension,
      kernel = nbKernel,
      stride = subsample,
      padding = paddingValue,
      activation = activation.doBuild(inputShape).asInstanceOf[TensorModule[T]],
      innerActivation = innerActivation.doBuild(inputShape).asInstanceOf[TensorModule[T]],
      wRegularizer = wRegularizer,
      uRegularizer = uRegularizer,
      bRegularizer = bRegularizer,
      withPeephole = false)
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
    borderMode: String = "valid",
    wRegularizer: Regularizer[T] = null,
    uRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    returnSequences: Boolean = false,
    goBackwards: Boolean = false,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): ConvLSTM2D[T] = {
    new ConvLSTM2D[T](nbFilter, nbKernel, KerasUtils.getKerasActivation(activation),
      KerasUtils.getKerasActivation(innerActivation),
      KerasUtils.toBigDLFormat5D(dimOrdering),
      subsample, borderMode, wRegularizer, uRegularizer, bRegularizer,
      returnSequences, goBackwards, inputShape)
  }
}
