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

import scala.reflect.ClassTag

/**
 * Stacks a list of n-dimensional tensors into one (n+1)-dimensional tensor.
 * @param dimension the dimension to stack along
 * @tparam T Numeric type. Only support float/double now
 */
@SerialVersionUID(3457313421501931556L)
class Pack[T: ClassTag] (val dimension: Int)(implicit ev: TensorNumeric[T])
  extends AbstractModule[Table, Tensor[T], T] {

  private def getPositiveDimension(input: Table): Int = {
    var nDim = this.dimension
    val firstInput: Tensor[T] = input(1)

    if (nDim < 0) {
      nDim = firstInput.dim() + nDim + 1
    }
    require(nDim <= firstInput.dim() + 1, "dimension exceeds input dimensions")
    nDim
  }

  override def updateOutput(input: Table): Tensor[T] = {
    val dimension = getPositiveDimension(input)

    val firstInput: Tensor[T] = input(1)
    val nDim = firstInput.nDimension()
    val size: Array[Int] = new Array[Int](nDim + 1)

    var i = 1
    while(i <= nDim + 1) {
      if (i < dimension) {
        size(i-1) = firstInput.size(i)
      } else if (i == dimension) {
        size(i-1) = input.length()
      } else {
        size(i-1) = firstInput.size(i - 1)
      }
      i = i + 1
    }


    output.resize(size)

    i = 1
    while (i <= input.length()) {
      val currentOutput: Tensor[T] = input(i)
      output.narrow(dimension, i, 1)
        .copy(currentOutput)
      i += 1
    }

    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]): Table = {
    val dimension = getPositiveDimension(input)

    var i = 1
    while (i <= input.length()) {
      if (!gradInput.contains(i)) {
        gradInput(i) = Tensor()
      }
      gradInput[Tensor[T]](i).resizeAs(input(i))
      i += 1
    }

    i = 1
    while (i <= input.length()) {
      val currentGradInput = gradOutput.select(dimension, i)
      gradInput[Tensor[T]](i).copy(currentGradInput)
      i += 1
    }
    gradInput
  }
}

object Pack {
  def apply[@specialized(Float, Double) T: ClassTag](
        dimension: Int)(implicit ev: TensorNumeric[T]): Pack[T] = {
    new Pack[T](dimension)
  }
}
