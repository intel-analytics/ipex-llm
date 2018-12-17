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

import com.intel.analytics.bigdl.nn.InferReshape
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, TensorModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape

import scala.reflect.ClassTag

/**
 * Reshapes an output to a certain shape.
 * Supports shape inference by allowing one -1 in the target shape.
 * For example, if inputShape = Shape(2, 3, 4), targetShape = Array(3, -1),
 * then outputShape will be Shape(3, 8).
 *
 * When you use this layer as the first layer of a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param targetShape Array of int. The target shape that you desire to have.
 *                    Batch dimension should be excluded.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class Reshape[T: ClassTag](
   val targetShape: Array[Int],
   val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasLayer.addBatch(inputShape)) {

  private var infer = false
  private var inferIndex = -1
  validateTargetShape()

  private def validateTargetShape(): Unit = {
    if (targetShape.contains(-1)) {
      infer = true
      var i = 0
      var inferCount = 0
      while (i < targetShape.length) {
        if (targetShape(i) == -1) {
          inferIndex = i
          inferCount += 1
        }
        // We don't consider 0 here, same as Keras
        else require(targetShape(i) >= 1,
          s"wrong reshape size at index $i: ${targetShape(i)}")
        i += 1
      }
      require(inferCount == 1, "can only specify one unknown dimension")
    }
  }

  override def computeOutputShape(inputShape: Shape): Shape = {
    val input = inputShape.toSingle().toArray
    val nonBatchInput = input.slice(1, input.length)
    if (infer) {
      val nElements = nonBatchInput.product
      val resizeElements = - targetShape.product
      require(nElements > resizeElements && nElements % resizeElements == 0,
      "Total size after reshape must be unchanged")
      targetShape(inferIndex) = nElements / resizeElements
    }
    else {
      require(targetShape.product == nonBatchInput.product,
        s"Total size after reshape must be unchanged. But In ${this.getName()} : " +
          s"original size is: ${ nonBatchInput.product }, " +
          s"reshape size is: ${ targetShape.product }")
    }
    Shape(Array(input(0)) ++ targetShape)
  }

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    var layer: TensorModule[T] = null
    if (infer) {
      layer = InferReshape(targetShape)
    }
    else {
      layer = com.intel.analytics.bigdl.nn.Reshape(targetShape, batchMode = Some(true))
    }
    layer.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object Reshape {
  def apply[@specialized(Float, Double) T: ClassTag](
    targetShape: Array[Int],
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): Reshape[T] = {
    new Reshape[T](targetShape, inputShape)
  }
}
