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
import com.intel.analytics.bigdl.tensor.TensorNumericMath.{NumericWildcard, TensorNumeric}
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

/**
 * Takes a table with two Tensor and returns the component-wise subtraction between them.
 */

@SerialVersionUID( - 7694575573537075609L)
class CSubTable[T: ClassTag]()(
  implicit ev: TensorNumeric[T]) extends AbstractModule[Table, Tensor[_], T]{

  override def updateOutput(input: Table): Tensor[_] = {
    val firstInput = input[Tensor[NumericWildcard]](1)
    if (output.getType() != firstInput.getType()) {
      output = firstInput.emptyInstance()
    }
    output.asInstanceOf[Tensor[NumericWildcard]].resizeAs(firstInput).copy(firstInput)
    output.asInstanceOf[Tensor[NumericWildcard]].sub(input[Tensor[NumericWildcard]](2))
    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[_]) : Table = {
    val firstInput = input[Tensor[NumericWildcard]](1)
    val secondInput = input[Tensor[NumericWildcard]](2)

    if (!gradInput.contains(1)) gradInput.insert(1, firstInput.emptyInstance())
    if (!gradInput.contains(2)) gradInput.insert(2, firstInput.emptyInstance())

    gradInput[Tensor[NumericWildcard]](1)
      .resizeAs(firstInput).copy(gradOutput.asInstanceOf[Tensor[NumericWildcard]])
    gradInput[Tensor[NumericWildcard]](2)
      .resizeAs(secondInput).copy(gradOutput.asInstanceOf[Tensor[NumericWildcard]])
      .mul(ev.fromType(-1))
    gradInput
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[CSubTable[T]]

  override def equals(other: Any): Boolean = other match {
    case that: CSubTable[T] =>
      super.equals(that) &&
        (that canEqual this)
    case _ => false
  }

  override def hashCode(): Int = {
    def getHashCode(a: Any): Int = if (a == null) 0 else a.hashCode()
    val state = Seq(super.hashCode())
    state.map(getHashCode).foldLeft(0)((a, b) => 31 * a + b)
  }
}

object CSubTable {
  def apply[@specialized(Float, Double) T: ClassTag]()
      (implicit ev: TensorNumeric[T]) : CSubTable[T] = {
    new CSubTable[T]()
  }
}
