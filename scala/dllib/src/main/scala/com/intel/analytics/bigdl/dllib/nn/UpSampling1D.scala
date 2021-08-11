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
import com.intel.analytics.bigdl.utils.Shape

import scala.reflect.ClassTag

/**
 * Upsampling layer for 1D inputs.
 * Repeats each temporal step length times along the time axis.
 *
 * If input's size is (batch, steps, features),
 * then the output's size is (batch, steps * length, features)
 *
 * @param length integer, upsampling factor.
 * @tparam T The numeric type in this module, usually which are [[Float]] or [[Double]]
 */
class UpSampling1D[T: ClassTag] (val length: Int)
  (implicit ev: TensorNumeric[T]) extends TensorModule[T] {
  require(length > 0, "UpSampling1D's length should be bigger than 0," +
    s"but got $length")

  override def computeOutputShape(inputShape: Shape): Shape = {
    val input = inputShape.toSingle().toArray
    require(input.length == 3,
      s"UpSampling1D requires 3D input, but got input dim ${input.length}")
    Shape(input(0), input(1) * length, input(2))
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim() == 3, "UpSampling1D only supports 3D input")
    require(input.isContiguous(), "input should be contiguous")

    val inputLength = input.size(3)
    val outputLength = inputLength * length

    output.resize(input.size(1), input.size(2) * length, input.size(3))

    val inputData = input.storage().array()
    val inputOffset = input.storageOffset() - 1

    val outputData = output.storage().array()
    val outputOffset = output.storageOffset() - 1

    var i = 0
    while (i < input.size(1) * input.size(2)) {
      var j = 0
      while (j < length) {
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
    val gradOutputLength = gradInputLength * length


    var i = 0
    while (i < input.size(1) * input.size(2)) {
      var j = 0
      while (j < length) {
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
  def apply[T: ClassTag](length: Int)
                        (implicit ev: TensorNumeric[T]): UpSampling1D[T] = {
    new UpSampling1D(length)
  }
}
