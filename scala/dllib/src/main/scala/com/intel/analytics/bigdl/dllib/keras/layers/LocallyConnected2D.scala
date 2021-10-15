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

import com.intel.analytics.bigdl.dllib.nn.keras.{KerasLayer, LocallyConnected2D => BigDLLocallyConnected2D}
import com.intel.analytics.bigdl.dllib.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.dllib.optim.Regularizer
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.Net
import com.intel.analytics.bigdl.dllib.keras.layers.utils.KerasUtils

import scala.reflect.ClassTag

/**
 * Locally-connected layer for 2D inputs that works similarly to the SpatialConvolution layer,
 * except that weights are unshared, that is, a different set of filters
 * is applied at each different patch of the input.
 * The input of this layer should be 4D.
 *
 * When using this layer as the first layer in a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param nbFilter Number of convolution filters to use.
 * @param nbRow Number of rows in the convolution kernel.
 * @param nbCol Number of columns in the convolution kernel.
 * @param activation Activation function to use. Default is null.
 *                   You can also pass in corresponding string representations such as 'relu'
 *                   or 'sigmoid', etc. for simple activations in the factory method.
 * @param borderMode Either 'valid' or 'same'. Default is 'valid'.
 * @param subsample Int array of length 2 corresponding to the step of the convolution in the height
 *                  and width dimension. Also called strides elsewhere. Default is (1, 1).
 * @param dimOrdering Format of input data. Either DataFormat.NCHW (dimOrdering='th') or
 *                    DataFormat.NHWC (dimOrdering='tf'). Default is NCHW.
 * @param wRegularizer An instance of [[Regularizer]], (eg. L1 or L2 regularization),
 *                     applied to the input weights matrices. Default is null.
 * @param bRegularizer An instance of [[Regularizer]], applied to the bias. Default is null.
 * @param bias Whether to include a bias (i.e. make the layer affine rather than linear).
 *             Default is true.
 * @param inputShape A Single Shape, does not include the batch dimension.
 * @tparam T The numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class LocallyConnected2D[T: ClassTag](
   override val nbFilter: Int,
   override val nbRow: Int,
   override val nbCol: Int,
   override val activation: KerasLayer[Tensor[T], Tensor[T], T] = null,
   override val borderMode: String = "valid",
   override val subsample: Array[Int] = Array(1, 1),
   override val dimOrdering: DataFormat = DataFormat.NCHW,
   wRegularizer: Regularizer[T] = null,
   bRegularizer: Regularizer[T] = null,
   override val bias: Boolean = true,
   override val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends BigDLLocallyConnected2D[T](nbFilter, nbRow, nbCol, activation, borderMode, subsample,
    dimOrdering, wRegularizer, bRegularizer, bias, inputShape) with Net {}

object LocallyConnected2D {
  def apply[@specialized(Float, Double) T: ClassTag](
    nbFilter: Int,
    nbRow: Int,
    nbCol: Int,
    activation: String = null,
    borderMode: String = "valid",
    subsample: (Int, Int) = (1, 1),
    dimOrdering: String = "th",
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    bias: Boolean = true,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): LocallyConnected2D[T] = {
    val subsampleArray = subsample match {
      case null => throw new IllegalArgumentException("For LocallyConnected2D, " +
        "subsample can not be null, please input int tuple of length 2")
      case _ => Array(subsample._1, subsample._2)
    }
    new LocallyConnected2D[T](nbFilter, nbRow, nbCol,
      KerasUtils.getKerasActivation(activation), borderMode, subsampleArray,
      KerasUtils.toBigDLFormat(dimOrdering), wRegularizer, bRegularizer, bias, inputShape)
  }
}
