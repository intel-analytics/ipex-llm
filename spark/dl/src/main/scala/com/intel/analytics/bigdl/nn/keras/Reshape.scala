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

class Reshape[T: ClassTag](
   val targetShape: Array[Int],
   var inputShape: Shape = null)(implicit ev: TensorNumeric[T])
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
      "total size after reshape must be unchanged")
      targetShape(inferIndex) = nElements / resizeElements
    }
    else {
      require(targetShape.product == nonBatchInput.product,
        s"total size after reshape must be unchanged. But In ${this.getName()} : " +
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
      layer = com.intel.analytics.bigdl.nn.Reshape(targetShape)
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
