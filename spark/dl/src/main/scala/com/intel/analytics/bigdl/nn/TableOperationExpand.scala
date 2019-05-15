/*
 * Copyright 2018 Analytics Zoo Authors.
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
 * @param operationLayer layer that can handle table operation, such as CSubTable, CMulTable, etc.
 * @param expandPos small tensor position in input table
 * @param ev$1
 * @param ev
 * @tparam T
 */
class TableOperationExpand[T: ClassTag](
   operationLayer: AbstractModule[Table, Tensor[T], T], expandPos: Int = 2)
   (implicit ev: TensorNumeric[T]) extends AbstractModule[Table, Tensor[T], T] {

  require(expandPos == 1 || expandPos == 2, s"TableOperationExpand:" +
    s"small tensor position in input table should be 1 or 2 , but get ${expandPos}")

  @transient
  private var expandLayer: AbstractModule[Tensor[T], Tensor[T], T] = null

  override def updateOutput(input: Table): Tensor[T] = {
    val inputSmall = input[Tensor[T]](expandPos)
    val inputLarge = input[Tensor[T]](3 - expandPos)

    if (expandLayer == null) expandLayer = ExpandSize(inputLarge.size())
    val inputExpand = expandLayer.forward(inputSmall)

    output = operationLayer.updateOutput(T(inputLarge, inputExpand))
    return output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]): Table = {
    val inputSmall = input[Tensor[T]](expandPos)
    val inputLarge = input[Tensor[T]](3 - expandPos)

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
  def apply[@specialized(Float, Double) T: ClassTag](expandPos: Int = 2)
    (implicit ev: TensorNumeric[T]) : TableOperationExpand[T] = {
    new TableOperationExpand[T](CMulTable[T], expandPos)
  }
}

object CSubTableExpand {
  def apply[@specialized(Float, Double) T: ClassTag](expandPos: Int = 2)
    (implicit ev: TensorNumeric[T]) : TableOperationExpand[T] = {
    new TableOperationExpand[T](CSubTable[T]
      .asInstanceOf[AbstractModule[Table, Tensor[T], T]], expandPos)
  }
}

