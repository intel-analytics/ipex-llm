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

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}

import scala.reflect.ClassTag

/**
 * Creates a module that takes a table as input and outputs the element at index `index`
 * (positive or negative). This can be either a table or a Tensor.
 * The gradients of the non-index elements are zeroed Tensors of the same size.
 * This is true regardless of the depth of the encapsulated Tensor as the function used
 * internally to do so is recursive.
 * @param index the index to be selected
 */
@SerialVersionUID(8787233248773612598L)
class SelectTable[T: ClassTag](
  val index: Int)
  (implicit ev: TensorNumeric[T]) extends AbstractModule[Table, Activity, T]  {

  override def updateOutput(input: Table): Activity = {
    val index = if (this.index < 0) input.length() + this.index else this.index

    require(input.contains(index), "index does not exist in the input table")
    output = input[Activity](index)

    output
  }

  override def updateGradInput(input: Table, gradOutput: Activity): Table = {
    Utils.zeroTableCopy(gradInput, input)
    val index = if (this.index < 0) {
      input.length() + this.index + 1
    } else {
      this.index
    }

    Utils.recursiveCopy(gradInput(index), gradOutput)

    require(gradInput.contains(index), "Index exceeds the size of input table")

    gradInput
  }

  override def toString: String = s"SelectTable($index)"


  override def canEqual(other: Any): Boolean = other.isInstanceOf[SelectTable[T]]

  override def equals(other: Any): Boolean = other match {
    case that: SelectTable[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        index == that.index
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(super.hashCode(), index)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }
}

object SelectTable {
  def apply[@specialized(Float, Double) T: ClassTag](
    dimension: Int)(implicit ev: TensorNumeric[T]) : SelectTable[T] = {
    new SelectTable[T](dimension)
  }
}
