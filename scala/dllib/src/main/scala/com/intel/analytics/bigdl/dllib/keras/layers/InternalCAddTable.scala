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

package com.intel.analytics.bigdl.dllib.keras.layers.internal

import com.intel.analytics.bigdl.dllib.nn.{CAddTable => BigDLCAddTable}
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.Table

import scala.reflect.ClassTag

class InternalCAddTable[T: ClassTag, D: ClassTag](override val inplace: Boolean = false)(
    implicit ev: TensorNumeric[T], ev2: TensorNumeric[D]) extends BigDLCAddTable[T, D](inplace) {

  private def canFastBroadcast(input: Tensor[D], gradOutput: Tensor[D]): Boolean = {
    require(input.dim() == gradOutput.dim(),
      s"input and gradOutput should have the same dims," +
        s"but got ${input.dim()} and ${gradOutput.dim()}")
    var i = 0
    while (i < input.size().length) {
      if ((i == 0 && input.size()(0) == 1) ||
        input.size()(i) == gradOutput.size()(i)) {
        i += 1
      } else {
        throw new IllegalArgumentException(s"input and gradOutput should have the same dims," +
          s"but got ${input.dim()} and ${gradOutput.dim()}")
      }
    }
    true
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[D]): Table = {
    var i = 1
    var sum = ev2.zero
    var calculateSum = false
    while (i <= input.length()) {
      if (i > gradInput.length) gradInput.insert(i, Tensor[T]().resizeAs(input(1)))
      if (inplace) {
        require(input[Tensor[D]](1).isSameSizeAs(gradOutput), "cannot use inplace for broadcast")
        gradInput[Tensor[D]](i).set(gradOutput)
      } else {
        if (input[Tensor[D]](i).isSameSizeAs(gradOutput)) {
          gradInput[Tensor[D]](i).resizeAs(gradOutput).copy(gradOutput)
        } else if (canFastBroadcast(input[Tensor[D]](i), gradOutput)) {
          gradInput[Tensor[D]](i).resizeAs(input[Tensor[D]](i)).copy(gradOutput.sum(1))
        } else {
          require(input[Tensor[D]](i).isScalar, "Only support scalar broadcast backward now")
          if (!calculateSum) {
            sum = gradOutput.sum()
            calculateSum = true
          }
          gradInput[Tensor[D]](i).resizeAs(input[Tensor[D]](i)).setValue(sum)
        }
      }
      i += 1
    }
    i = input.length + 1
    while (i <= gradInput.length) {
      gradInput.remove(i)
    }
    gradInput
  }

}
