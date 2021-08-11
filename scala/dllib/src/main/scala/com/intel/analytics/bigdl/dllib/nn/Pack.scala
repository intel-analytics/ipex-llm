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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.{NumericWildcard, TensorNumeric}
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

/**
 * Stacks a list of n-dimensional tensors into one (n+1)-dimensional tensor.
 * @param dimension the dimension to stack along
 * @tparam T Numeric type. Only support float/double now
 */
@SerialVersionUID(3457313421501931556L)
class Pack[T: ClassTag] (val dimension: Int)(implicit ev: TensorNumeric[T])
  extends AbstractModule[Activity, Tensor[_], T] {

  private def getPositiveDimension(input: Table): Int = {
    var nDim = this.dimension
    val firstInput: Tensor[_] = input(1)

    if (nDim < 0) {
      nDim = firstInput.dim() + nDim + 1
    }
    require(nDim <= firstInput.dim() + 1, "dimension exceeds input dimensions" +
      s"dimension $nDim, inputDimension ${firstInput.dim()}")
    nDim
  }

  override def updateOutput(input: Activity): Tensor[_] = {

    val tableInput = input match {
      case t: Tensor[_] => T(t)
      case t: Table => t
    }

    val dimension = getPositiveDimension(tableInput)

    val firstInput: Tensor[_] = tableInput(1)
    val nDim = firstInput.nDimension()
    val size: Array[Int] = new Array[Int](nDim + 1)

    var i = 1
    while(i <= nDim + 1) {
      if (i < dimension) {
        size(i-1) = firstInput.size(i)
      } else if (i == dimension) {
        size(i-1) = tableInput.length()
      } else {
        size(i-1) = firstInput.size(i - 1)
      }
      i = i + 1
    }

    if (output.getType() != firstInput.getType()) {
      output = firstInput.emptyInstance()
    }

    output.resize(size)

    i = 1
    while (i <= tableInput.length()) {
      val currentOutput = tableInput[Tensor[NumericWildcard]](i)
      output.narrow(dimension, i, 1).asInstanceOf[Tensor[NumericWildcard]]
        .copy(currentOutput)
      i += 1
    }

    output
  }

  override def updateGradInput(input: Activity, gradOutput: Tensor[_]): Activity = {
    val tableInput = input match {
      case t: Tensor[_] => T(t)
      case t: Table => t
    }
    val dimension = getPositiveDimension(tableInput)

    val firstInput = tableInput[Tensor[_]](1)

    if (input.isTensor) {
      if (gradInput == null ||
        gradInput.asInstanceOf[Tensor[_]].getType() != firstInput.getType()) {
        gradInput = firstInput.emptyInstance()
      }
      val gradInputTensor = gradInput.asInstanceOf[Tensor[NumericWildcard]]
      gradInputTensor.resizeAs(firstInput)
      gradInputTensor.copy(firstInput.asInstanceOf[Tensor[NumericWildcard]])
    } else {
      if (gradInput == null) gradInput = T()
      val gradInputTable = gradInput.toTable
      var i = 1
      while (i <= tableInput.length()) {
        if (!gradInputTable.contains(i)) {
          gradInputTable(i) = gradOutput.emptyInstance()
        }
        gradInputTable[Tensor[_]](i).resizeAs(tableInput(i))
        i += 1
      }

      i = 1
      while (i <= tableInput.length()) {
        val currentGradInput = gradOutput.select(dimension, i).asInstanceOf[Tensor[NumericWildcard]]
        gradInputTable[Tensor[NumericWildcard]](i).copy(currentGradInput)
        i += 1
      }
    }

    gradInput
  }
}

object Pack {
  def apply[T: ClassTag](
        dimension: Int)(implicit ev: TensorNumeric[T]): Pack[T] = {
    new Pack[T](dimension)
  }
}
