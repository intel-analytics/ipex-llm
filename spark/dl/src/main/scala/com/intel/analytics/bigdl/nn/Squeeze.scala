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
 * Delete singleton all dimensions or a specific dim.
 *
 * @param dim Optional. The dimension to be delete. Default: delete all dimensions.
 * @param numInputDims Optional. If in a batch model, set to the inputDims.
 */

@SerialVersionUID(7998127436291978408L)
class Squeeze[@specialized(Float, Double) T: ClassTag](
  var dim : Int = Int.MinValue,
  var numInputDims: Int = Int.MinValue
  )(implicit ev: TensorNumeric[T]) extends TensorModule[T]  {

  def setNumInputDims(numInputDims: Int): Unit = {
    this.numInputDims = numInputDims
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    var addOne = false
    if (numInputDims != Int.MinValue && input.dim() == numInputDims + 1) {
      if (dim != Int.MinValue) {
        dim += 1
      } else if (input.size(1) == 1) {
        addOne = true // in case of miniBatch of size 1
      }
    }
    output.set(input)
    if (dim != Int.MinValue) output.squeeze(dim) else output.squeeze()
    if (addOne) {
      val s = output.size()
      s(1) = 1
      output.set(output.view(s))
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    require(input.nElement() == gradOutput.nElement())
    gradInput.set(gradOutput.view(input.size()))
    gradInput
  }

  override def toString(): String = {
    s"nn.Squeeze(${if (dim != Int.MinValue) dim + ", " else ""}" +
      s"${if (numInputDims != Int.MinValue) numInputDims else ""})"
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[Squeeze[T]]

  override def equals(other: Any): Boolean = other match {
    case that: Squeeze[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        dim == that.dim &&
        numInputDims == that.numInputDims
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(super.hashCode(), dim, numInputDims)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }
}

object Squeeze {
  def apply[@specialized(Float, Double) T: ClassTag](
      dim : Int = Int.MinValue,
      numInputDims: Int = Int.MinValue)(implicit ev: TensorNumeric[T]) : Squeeze[T] = {
    new Squeeze[T](dim, numInputDims)
  }
}
