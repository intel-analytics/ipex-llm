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

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.augmentation.Expand
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

/**
 * When two tensors have different size, firstly expand small size tensor to large size tensor,
 * and then do table operation.
 * @param operationLayer layer that handles table operation, such as CSubTable, CMulTable, etc.
 * @param ev$1
 * @param ev
 * @tparam T
 */
class TableOperation[T: ClassTag](
   val operationLayer: AbstractModule[Table, Tensor[T], T])
   (implicit ev: TensorNumeric[T]) extends AbstractModule[Table, Tensor[T], T] {

  @transient
  private var expandLayer: AbstractModule[Tensor[T], Tensor[T], T] = null

  // small tensor position in input table
  private var smallPos = 1

  override def updateOutput(input: Table): Tensor[T] = {
    // only support table with two tensors
    require(input.length() == 2, s"Only support input two tensors, but get ${input.length()}")
    // get small tensor position in table
    val input1 = input[Tensor[T]](1)
    val input2 = input[Tensor[T]](2)
    if (input1.nElement() > input2.nElement()) {
      smallPos = 2
    }

    val inputSmall = input[Tensor[T]](smallPos)
    val inputLarge = input[Tensor[T]](3 - smallPos)

    val largeSize = inputLarge.size()
    // batchSize may be not same for model inference
    largeSize(0) = -1
    if (expandLayer == null) expandLayer = ExpandSize(largeSize)
    val inputExpand = expandLayer.forward(inputSmall)

    output = operationLayer.updateOutput(T(inputLarge, inputExpand))
    return output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]): Table = {
    val inputSmall = input[Tensor[T]](smallPos)
    val inputLarge = input[Tensor[T]](3 - smallPos)

    val inputExpand = expandLayer.output
    gradInput = operationLayer.updateGradInput(T(inputLarge, inputExpand), gradOutput)
    gradInput(2) = expandLayer.backward(inputSmall, gradInput[Tensor[T]](2))
    gradInput
  }

  override def toString: String = s"TableOperationExpand"

  override def clearState(): this.type = {
    if (expandLayer != null) expandLayer.clearState()
    operationLayer.clearState()
    this
  }
}

object CMulTableExpand {
  def apply[@specialized(Float, Double) T: ClassTag]()
    (implicit ev: TensorNumeric[T]) : TableOperation[T] = {
    new TableOperation[T](CMulTable[T])
  }
}

object CSubTableExpand {
  def apply[@specialized(Float, Double) T: ClassTag]()
    (implicit ev: TensorNumeric[T]) : TableOperation[T] = {
    new TableOperation[T](CSubTable[T]
      .asInstanceOf[AbstractModule[Table, Tensor[T], T]])
  }
}

