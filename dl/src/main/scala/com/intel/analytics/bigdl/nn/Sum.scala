/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
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
 * It is a simple layer which applies a sum operation over the given dimension.
 * When nInputDims is provided, the input will be considered as a batches.
 * Then the sum operation will be applied in (dimension + 1)
 *
 * @param dimension the dimension to be applied sum operation
 * @param nInputDims the number of dimensions of the give input
 * @param sizeAverage default is false, if it is true, it will return the mean instead
 */

@SerialVersionUID(- 8025422596092583688L)
class Sum[T: ClassTag](
  dimension: Int = 1,
  nInputDims: Int = -1,
  sizeAverage: Boolean = false)
  (implicit ev: TensorNumeric[T]) extends TensorModule[T] {
  @transient
  private var _gradOutput: Tensor[T] = null

  private def getPositiveDimension(input: Tensor[T]): Int = {
    var dimension = this.dimension
    if (dimension < 0) {
      dimension = input.dim() + dimension + 1
    } else if (nInputDims > 0 && input.dim() == (nInputDims + 1)) {
      dimension += 1
    }
    require(input.dim() >= dimension, "dimension exceeds input dimensions")
    dimension
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    val dimension = getPositiveDimension(input)
    output.sum(input, dimension)

    if (sizeAverage) {
      output.div(ev.fromType[Int](input.size(dimension)))
    }
    if (output.nDimension() > 1) {
      output.set(output.select(dimension, 1))
    }

    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    val dimension = getPositiveDimension(input)
    val size = input.size()
    size(dimension - 1) = 1

    if (!gradOutput.isContiguous()) {
      _gradOutput = gradOutput.clone().view(size)
    } else {
      _gradOutput = gradOutput.view(size)
    }
    gradInput.resizeAs(input)
    gradInput.copy(_gradOutput.expandAs(input))
    if (sizeAverage) {
      gradInput.div(ev.fromType[Int](input.size(dimension)))
    }
    gradInput
  }

  override def toString: String = s"nn.Sum"
}

object Sum {
  def apply[@specialized(Float, Double) T: ClassTag](
      dimension: Int = 1,
      nInputDims: Int = -1,
      sizeAverage: Boolean = false)(implicit ev: TensorNumeric[T]) : Sum[T] = {
    new Sum[T](dimension, nInputDims, sizeAverage)
  }
}
