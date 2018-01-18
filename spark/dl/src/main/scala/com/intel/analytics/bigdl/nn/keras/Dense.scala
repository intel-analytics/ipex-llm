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

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn._
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape

import scala.reflect.ClassTag


@SerialVersionUID( 359656776803598944L)
class Dense[T: ClassTag](val outputDim: Int,
                         val init: InitializationMethod = RandomUniform,
                         val activation: TensorModule[T] = null,
                         var wRegularizer: Regularizer[T] = null,
                         var bRegularizer: Regularizer[T] = null,
                         val bias: Boolean = true,
                         var inputShape: Array[Int] = null
  )(implicit ev: TensorNumeric[T]) extends KerasLayer[Tensor[T], Tensor[T], T](Shape(inputShape)) {

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val layer = Linear(
      inputSize = inputShape.toSingle()(1), // the first dim is batch
      outputSize = outputDim,
      withBias = bias,
      wRegularizer = wRegularizer,
      bRegularizer = bRegularizer
    )
    layer.setInitMethod(weightInitMethod = init, biasInitMethod = Zeros)
    KerasLayer.fuse(layer,
      activation,
      inputShape).asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object Dense {

  def apply[@specialized(Float, Double) T: ClassTag](
    outputDim: Int,
    init: InitializationMethod = RandomUniform,
    activation: TensorModule[T] = null,
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    bias: Boolean = true,
    inputShape: Array[Int] = null)(implicit ev: TensorNumeric[T]) : Dense[T] = {
      new Dense[T](
        outputDim,
        init,
        activation,
        wRegularizer,
        bRegularizer,
        bias,
        inputShape)
  }
}


