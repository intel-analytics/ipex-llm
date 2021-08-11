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

import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * [[Masking]] Use a mask value to skip timesteps for a sequence
 *
 * @param maskValue mask value
 */
class Masking[T: ClassTag](maskValue: Double = 0.0)
(implicit ev: TensorNumeric[T]) extends TensorModule[T]{
  val batchDim = 1
  val timeDim = 2

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output.resizeAs(input)
    var timeIndex = 1
    var batchIndex = 1
    val fillValue = ev.fromType(0.0)
    while(batchIndex <= input.size(batchDim)) {
      val batchInput = input.select(batchDim, batchIndex)
      val batchOutput = output.select(batchDim, batchIndex)
      while(timeIndex <= input.size(timeDim)) {
        val slicedTensor = batchInput.select(timeDim - 1, timeIndex)
        if (!slicedTensor.notEqualValue(maskValue)) {
          batchOutput.select(timeDim - 1, timeIndex).fill(fillValue)
        } else {
          batchOutput.select(timeDim - 1, timeIndex).copy(slicedTensor)
        }
        timeIndex += 1
      }
      batchIndex += 1
      timeIndex = 1
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    require(input.isSameSizeAs(gradOutput),
      "Input should have the same size as gradOutput" +
        s"input size(${input.size().foreach(x => x)})" +
        s"gradOutput size(${gradOutput.size().foreach(x => x)})")
    gradInput.resizeAs(input)
    var timeIndex = 1
    var batchIndex = 1
    val fillValue = ev.fromType(0.0)
    while(batchIndex <= input.size(batchDim)) {
      val batchInput = input.select(batchDim, batchIndex)
      val batchgradOutput = gradOutput.select(batchDim, batchIndex)
      val batchgradInput = gradInput.select(batchDim, batchIndex)
      while(timeIndex <= input.size(timeDim)) {
        val slicedTensor = batchInput.select(timeDim - 1, timeIndex)
        if (!slicedTensor.notEqualValue(maskValue)) {
          batchgradInput.select(timeDim - 1, timeIndex).fill(fillValue)
        } else {
          batchgradInput.select(timeDim - 1, timeIndex).copy(
            batchgradOutput.select(timeDim - 1, timeIndex))
        }
        timeIndex += 1
      }
      batchIndex += 1
      timeIndex = 1
    }
    gradInput
  }
}

object Masking {
  def apply[T : ClassTag](maskValue: Double)(implicit ev: TensorNumeric[T]): Masking[T]
    = new Masking[T](maskValue)
}
