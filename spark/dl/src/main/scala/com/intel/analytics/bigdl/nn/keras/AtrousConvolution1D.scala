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
import com.intel.analytics.bigdl.nn.{InitializationMethod, Xavier, Zeros}
import com.intel.analytics.bigdl.nn.{SpatialDilatedConvolution, Squeeze, Transpose, Sequential => TSequential}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape

import scala.reflect.ClassTag

class AtrousConvolution1D[T: ClassTag](
   val nbFilter: Int,
   val filterLength: Int,
   val init: InitializationMethod = Xavier,
   val activation: AbstractModule[Tensor[T], Tensor[T], T] = null,
   val subsampleLength: Int = 1,
   val atrousRate: Int = 1,
   var wRegularizer: Regularizer[T] = null,
   var bRegularizer: Regularizer[T] = null,
   val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasLayer.addBatch(inputShape)) {

  override def computeOutputShape(inputShape: Shape): Shape = {
    val input = inputShape.toSingle().toArray
    require(input.length == 3,
      s"AtrousConvolution1D requires 3D input, but got input dim ${input.length}")
    val length = KerasUtils.computeConvOutputLength(input(1), filterLength,
      "valid", subsampleLength, atrousRate)
    Shape(input(0), length, nbFilter)
  }

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val input = inputShape.toSingle().toArray
    val model = TSequential[T]()
    model.add(Transpose(Array((2, 3))))
    model.add(com.intel.analytics.bigdl.nn.Reshape(Array(input(2), input(1), 1), Some(true)))
    val layer = SpatialDilatedConvolution(
      nInputPlane = input(2),
      nOutputPlane = nbFilter,
      kW = 1,
      kH = filterLength,
      dW = 1,
      dH = subsampleLength,
      dilationW = 1,
      dilationH = atrousRate,
      wRegularizer = wRegularizer,
      bRegularizer = bRegularizer)
    layer.setInitMethod(weightInitMethod = init, biasInitMethod = Zeros)
    model.add(layer)
    model.add(Transpose(Array((2, 3))))
    model.add(Squeeze(4))
    if (activation != null) {
      model.add(activation)
    }
    model.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object AtrousConvolution1D {
  def apply[@specialized(Float, Double) T: ClassTag](
    nbFilter: Int,
    filterLength: Int,
    init: String = "glorot_uniform",
    activation: String = null,
    subsampleLength: Int = 1,
    atrousRate: Int = 1,
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): AtrousConvolution1D[T] = {
    new AtrousConvolution1D[T](nbFilter, filterLength, KerasUtils.getInitMethod(init),
      KerasUtils.getActivation(activation), subsampleLength, atrousRate,
      wRegularizer, bRegularizer, inputShape)
  }
}
