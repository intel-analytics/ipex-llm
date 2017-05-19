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
 * Transpose input along specified dimensions
 * @param permutations dimension pairs that need to swap
 */
@SerialVersionUID(8543726779794064339L)
class Transpose[@specialized(Float, Double) T: ClassTag](
  val permutations: Array[(Int, Int)])(implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  var buffer: Tensor[T] = _

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    var i = 0
    buffer = input
    while (i < permutations.length) {
      buffer = buffer.transpose(permutations(i)._1, permutations(i)._2)
      i += 1
    }
    output.resizeAs(buffer).copy(buffer)
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    var i = permutations.length - 1
    buffer = gradOutput
    while (i >= 0) {
      buffer = buffer.transpose(permutations(i)._1, permutations(i)._2)
      i -= 1
    }
    gradInput.resizeAs(buffer).copy(buffer)
    gradInput
  }

  override def toString(): String = {
    s"${getPrintName}(${
      permutations.map {
        case (from: Int, to: Int) => s"$from -> $to"
      }.mkString(", ")
    })"
  }
}

object Transpose {
  def apply[@specialized(Float, Double) T: ClassTag](
      permutations: Array[(Int, Int)])(implicit ev: TensorNumeric[T]) : Transpose[T] = {
    new Transpose[T](permutations)
  }
}
