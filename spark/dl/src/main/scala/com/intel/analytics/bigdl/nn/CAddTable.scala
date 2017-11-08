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
import com.intel.analytics.bigdl.utils.Table

import scala.reflect._

/**
 * Merge the input tensors in the input table by element wise adding them together. The input table
 * is actually an array of tensor with same size.
 * @param inplace reuse the input memory
 * @param ev numeric operator
 * @tparam T Numeric type. Only support float/double now
 */
@SerialVersionUID(7959261460060075605L)
class CAddTable[T: ClassTag](val inplace: Boolean = false)(
  implicit ev: TensorNumeric[T]) extends AbstractModule[Table, Tensor[T], T] {

  override def updateOutput(input: Table): Tensor[T] = {
    var scalar = ev.zero
    var hasTensor = false
    var hasScalar = false
    var initTensor = false

    var i = 1
    while (i <= input.length()) {
      val curTensor = input[Tensor[T]](i)
      if (curTensor.isScalar) {
        scalar = ev.plus(scalar, curTensor.value())
        hasScalar = true
      } else if (curTensor.isTensor) {
        if (initTensor) {
          output = output.add(curTensor)
        } else {
          if (inplace) {
            output.set(curTensor)
          } else {
            output.resizeAs(curTensor).copy(curTensor)
          }
          initTensor = true
        }
        hasTensor = true
      }
      i += 1
    }

    if (hasTensor && hasScalar) {
      output.add(scalar)
    } else if (hasScalar) {
      if (inplace) {
        output.set(input[Tensor[T]](1)).setValue(scalar)
      } else {
        output.resizeAs(input[Tensor[T]](1)).setValue(scalar)
      }
    }

    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]) : Table = {
    var i = 1
    var sum = ev.zero
    var calculateSum = false
    while (i <= input.length()) {
      if (i > gradInput.length) gradInput.insert(i, Tensor[T]().resizeAs(input(1)))
      if (inplace) {
        require(input[Tensor[T]](1).isSameSizeAs(gradOutput), "cannot use inplace for broadcast")
        gradInput[Tensor[T]](i).set(gradOutput)
      } else {
        if (input[Tensor[T]](i).isSameSizeAs(gradOutput)) {
          gradInput[Tensor[T]](i).resizeAs(gradOutput).copy(gradOutput)
        } else {
          require(input[Tensor[T]](i).isScalar, "Only support scalar broadcast backward now")
          if (!calculateSum) {
            sum = gradOutput.sum()
            calculateSum = true
          }
          gradInput[Tensor[T]](i).resizeAs(input[Tensor[T]](i)).setValue(sum)
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

  override def clearState(): this.type = {
    if (!inplace) {
      super.clearState()
    }
    this
  }
}


object CAddTable {
  def apply[@specialized(Float, Double) T: ClassTag](
      inplace: Boolean = false)(implicit ev: TensorNumeric[T]) : CAddTable[T] = {
    new CAddTable[T](inplace)
  }
}


