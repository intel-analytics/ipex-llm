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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, DataFormat}
import com.intel.analytics.bigdl.nn.{Squeeze, Sequential => TSequential}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape

import scala.reflect.ClassTag

class LocallyConnected1D[T: ClassTag](
   val nbFilter: Int,
   val filterLength: Int,
   val activation: AbstractModule[Tensor[T], Tensor[T], T] = null,
   val subsampleLength: Int = 1,
   var wRegularizer: Regularizer[T] = null,
   var bRegularizer: Regularizer[T] = null,
   val bias: Boolean = true,
   val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasLayer.addBatch(inputShape)) {

  override def computeOutputShape(inputShape: Shape): Shape = {
    val input = inputShape.toSingle().toArray
    require(input.length == 3,
      s"LocallyConnected1D requires 3D input, but got input dim ${input.length}")
    val length = KerasUtils.computeConvOutputLength(input(1), filterLength,
      "valid", subsampleLength)
    Shape(input(0), length, nbFilter)
  }

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val input = inputShape.toSingle().toArray
    val model = TSequential[T]()
    model.add(com.intel.analytics.bigdl.nn.Reshape(Array(input(1), 1, input(2)), Some(true)))
    val layer = com.intel.analytics.bigdl.nn.LocallyConnected2D(
      nInputPlane = input(2),
      inputWidth = 1,
      inputHeight = input(1),
      nOutputPlane = nbFilter,
      kernelW = 1,
      kernelH = filterLength,
      strideW = 1,
      strideH = subsampleLength,
      wRegularizer = wRegularizer,
      bRegularizer = bRegularizer,
      withBias = bias,
      format = DataFormat.NHWC)
    model.add(layer)
    model.add(Squeeze(3))
    if (activation != null) {
      model.add(activation)
    }
    model.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object LocallyConnected1D {
  def apply[@specialized(Float, Double) T: ClassTag](
    nbFilter: Int,
    filterLength: Int,
    activation: String = null,
    subsampleLength: Int = 1,
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    bias: Boolean = true,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): LocallyConnected1D[T] = {
    new LocallyConnected1D[T](nbFilter, filterLength,
      KerasUtils.getActivation(activation), subsampleLength,
      wRegularizer, bRegularizer, bias, inputShape)
  }
}
