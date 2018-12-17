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
 * Tile repeats input `nFeatures` times along its `dim` dimension
 *
 *
 * @param dim dimension to be replicated.
 * @param copies specify the number of copies.
 */

@SerialVersionUID( - 7341965298635163982L)
class Tile[T: ClassTag](
  val dim : Int = 1,
  val copies : Int = 2)
    (implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  require(dim > 0, "Can only replicate across positive integer dimensions.")
  require(copies >= 2, "copies should be at least 2")

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(dim <= input.dim() + 1,
      s"Not enough input dimensions to replicate along dimension $dim.")

    val sizes = new Array[Int](input.size().length)

    var index = 0

    input.size().foreach(size => {
      index += 1
      sizes(index - 1) = if (index == dim) copies else 1
    })

    output = input.repeatTensor(sizes)
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(output).zero()
    val size = new Array[Int](input.dim() + 1)
    var i = 0
    while (i < input.dim()) {
      size(i) = input.size(i)
      i += 1
    }
    size(dim - 1) = size(dim - 1) * copies
    gradInput.view(size).sum(gradOutput, dim)
    gradInput
  }

  override def toString(): String = {
    s"${getPrintName}($dim,$copies)"
  }

}

object Tile {
  def apply[@specialized(Float, Double) T: ClassTag](
    dim : Int = 1,
    copies : Int = 2)(implicit ev: TensorNumeric[T]) : Tile[T] = {
    new Tile[T](dim, copies)
  }
}
