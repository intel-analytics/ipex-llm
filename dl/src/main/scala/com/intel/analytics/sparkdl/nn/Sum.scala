/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
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
package com.intel.analytics.sparkdl.nn

import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class Sum[T: ClassTag](
  dimension: Int = 1,
  nInputDims: Int = -1,
  sizeAverage: Boolean = false)
  (implicit ev: TensorNumeric[T]) extends TensorModule[T] {
  @transient
  var _gradOutput: Tensor[T] = null

  def getPositiveDimension(input: Tensor[T]): Int = {
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
      if (_gradOutput == null) {
        _gradOutput = Tensor[T]()
      }
      _gradOutput
        .resizeAs(gradOutput)
        .copy(gradOutput)
        .view(size)
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

  override def toString(): String = {
    s"nn.Tanh"
  }
}
