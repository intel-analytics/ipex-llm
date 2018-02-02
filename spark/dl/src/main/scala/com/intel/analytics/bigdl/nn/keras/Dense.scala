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

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn._
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape

import scala.reflect.ClassTag

class Dense[T: ClassTag](
   val outputDim: Int,
   val init: InitializationMethod = Xavier,
   val activation: AbstractModule[Tensor[T], Tensor[T], T] = null,
   var wRegularizer: Regularizer[T] = null,
   var bRegularizer: Regularizer[T] = null,
   val bias: Boolean = true,
   var inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasLayer.addBatch(inputShape)) {

  override def computeOutputShape(inputShape: Shape): Shape = {
    val input = inputShape.toSingle().toArray
    require(inputShape.toSingle().size >=2,
      s"Dense requires input dim >=2, but got dim: ${inputShape.toSingle().length}")
    Shape(input.slice(0, input.length -1) ++ Array(outputDim))
  }

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val inputShapeList = inputShape.toSingle()
    var layer = Linear(
      inputSize = inputShapeList.last,
      outputSize = outputDim,
      withBias = bias,
      wRegularizer = wRegularizer,
      bRegularizer = bRegularizer)
    layer.setInitMethod(weightInitMethod = init, biasInitMethod = Zeros)

    if (inputShape.toSingle().size <= 2) {
      KerasLayer.fuse(layer, activation,
        inputShape).asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
    } else {
      val seq = new Sequential[T](stopInferShape = true)
      val inDim = inputShapeList.last
      seq.add(InputLayer(inputShape = inputShape))
      seq.add(InferReshape(Array(-1, inDim), false))
      seq.add(layer)
      seq.add(InferReshape(Array(-1) ++
        inputShapeList.slice(1, inputShapeList.size - 1) ++ Array(outputDim), false))
      if (activation != null) {
        seq.add(activation)
      }
      seq.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
    }
  }
}

object Dense {
  def apply[@specialized(Float, Double) T: ClassTag](
    outputDim: Int,
    init: String = "glorot_uniform",
    activation: String = null,
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    bias: Boolean = true,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): Dense[T] = {
    new Dense[T](outputDim, KerasUtils.getInitMethod(init),
      KerasUtils.getActivation(activation),
      wRegularizer, bRegularizer, bias, inputShape)
  }
}


