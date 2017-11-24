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

import java.util

import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class UpSampling1D[T: ClassTag] (size: Int)
  (implicit ev: TensorNumeric[T]) extends TensorModule[T] {
  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim() == 3, "UpSampling1D only supports 3D input")
    require(input.isContiguous(), "input should be contiguous")

    val inputLength = input.size(3)
    val outputLength = inputLength * size

    output.resize(input.size(1), input.size(2) * size, input.size(3))

    val inputData = input.storage().array()
    val inputOffset = input.storageOffset() - 1

    val outputData = output.storage().array()
    val outputOffset = output.storageOffset() - 1

    var i = 0
    while (i < input.size(1) * input.size(2)) {
      var j = 0
      while (j < size) {
        ev.arraycopy(inputData, inputOffset + i * inputLength,
          outputData, outputOffset + i * outputLength + inputLength * j, inputLength)
        j += 1
      }
      i += 1
    }

    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    require(gradOutput.dim() == 3, "UpSampling1D only supports 3D input")
    require(gradOutput.isContiguous(), "gradOutput should be contiguous")
    gradInput.resizeAs(input).zero()

    val gradInputData = gradInput.storage().array()
    val gradInputOffset = gradInput.storageOffset() - 1

    val gradOutputData = gradOutput.storage().array()
    val gradOutputOffset = gradOutput.storageOffset() - 1

    val gradInputLength = gradInput.size(3)
    val gradOutputLength = gradInputLength * size


    var i = 0
    while (i < input.size(1) * input.size(2)) {
      var j = 0
      while (j < size) {
        ev.axpy(gradInputLength, ev.one, gradOutputData,
          gradOutputOffset + i * gradOutputLength + gradInputLength * j, 1,
          gradInputData, gradInputOffset + i * gradInputLength, 1)
        j += 1
      }
      i += 1
    }

    gradInput
  }
}

object UpSampling1D {
  def apply[T: ClassTag](size: Int)
                        (implicit ev: TensorNumeric[T]): UpSampling1D[T] = {
    new UpSampling1D(size)
  }
}