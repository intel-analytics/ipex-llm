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
import com.intel.analytics.bigdl.nn.abstractnn._
import com.intel.analytics.bigdl.nn.{Sequential => TSequential, _}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape

import scala.reflect.ClassTag


class Dense[T: ClassTag](val outputDim: Int,
    val init: InitializationMethod = RandomUniform,
    val activation: TensorModule[T] = null,
    var wRegularizer: Regularizer[T] = null,
    var bRegularizer: Regularizer[T] = null,
    val bias: Boolean = true,
    var inputShape: Shape = null
)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasLayer.addBatch(inputShape)) {

  override def computeOutputShape(inputShape: Shape): Shape = {
    require(inputShape.toSingle().size >=2,
      s"inputShape should at least containing 2 dims, but got: $inputShape.toSingle().size")
    inputShape.copyAndUpdate(-1, outputDim)
  }

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val inputShapeList = inputShape.toSingle()
    var model: AbstractModule[Tensor[T], Tensor[T], T] = Linear(
      inputSize = inputShapeList.last,
      outputSize = outputDim,
      withBias = bias,
      wRegularizer = wRegularizer,
      bRegularizer = bRegularizer
    ).setInitMethod(weightInitMethod = init, biasInitMethod = Zeros)

    model = KerasLayer.fuse(model,
      activation,
      inputShape).asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]

    if (inputShape.toSingle().size <= 2) {
      model
    } else {
      val seq = new Sequential[T](stopInferShape = true)
      val inDim = inputShapeList.last
      seq.add(InputLayer(inputShape = inputShape))
      seq.add(InferReshape(Array(-1, inDim), false))
      seq.add(model)
      seq.add(InferReshape(Array(-1) ++
        inputShapeList.slice(1, inputShapeList.size - 1) ++ Array(outputDim), false))
      seq
    }.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
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
      inputShape: Shape = null)(implicit ev: TensorNumeric[T]) : Dense[T] = {
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


