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
import com.intel.analytics.bigdl.nn.Maxout
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape

import scala.reflect.ClassTag

class MaxoutDense[T: ClassTag](
   val outputDim: Int,
   val nbFeature: Int = 4,
   val wRegularizer: Regularizer[T] = null,
   var bRegularizer: Regularizer[T] = null,
   val bias: Boolean = true,
   val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasLayer.addBatch(inputShape)) {

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val input = inputShape.toSingle().toArray
    val layer = Maxout(
      inputSize = input(1),
      outputSize = outputDim,
      maxoutNumber = nbFeature,
      withBias = bias,
      wRegularizer = wRegularizer,
      bRegularizer = bRegularizer)
    layer.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object MaxoutDense {
  def apply[@specialized(Float, Double) T: ClassTag](
    outputDim: Int,
    nbFeature: Int = 4,
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    bias: Boolean = true,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): MaxoutDense[T] = {
    new MaxoutDense[T](outputDim, nbFeature, wRegularizer, bRegularizer, bias, inputShape)
  }
}
