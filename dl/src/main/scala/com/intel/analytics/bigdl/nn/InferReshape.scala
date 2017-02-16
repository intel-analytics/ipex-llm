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

import breeze.linalg.sum
import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * Reshape with the support of infered size, the infered dim is indicated as -1
 * note that at most one infered dim is supported,
 * the element number is not changed after reshape
 * e.g. reshape from (3, 2) to (2, -1) is reshape to (2, 3)
 * @param size      the target tensor size
 * @param batchMode whether in batch mode
 * @tparam T type
 */
class InferReshape[@specialized(Float, Double) T: ClassTag](
  size: Array[Int], var batchMode: Boolean = false)(
  implicit ev: TensorNumeric[T]) extends TensorModule[T] {
  val inferedSizes = if (batchMode) new Array[Int](size.length + 1) else new Array[Int](size.length)

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    if (size.contains(-1)) {
      assert(sum(size.map(x => if (x == ev.fromType(-1)) 1 else 0)) == 1,
        "at most a single (1) value of -1 may be specified")
    }
    var inferIndex = -1
    var count = 1
    val startIndex = if (batchMode) 1 else 0
    var i = 0
    while (i < size.length) {
      if (size(i) == 0) {
        inferedSizes(i + startIndex) = input.size(i + startIndex + 1 - startIndex)
        count *= inferedSizes(i + startIndex)
      }
      else if (size(i) == -1) inferIndex = i + startIndex
      else {
        inferedSizes(i + startIndex) = size(i)
        count *= size(i)
      }
      i += 1
    }
    require(count <= input.nElement())
    if (inferIndex != -1) {
      inferedSizes(inferIndex) = input.nElement() / count
      if (batchMode) inferedSizes(inferIndex) = inferedSizes(inferIndex) / input.size(1)
    }

    if (batchMode) {
      inferedSizes(0) = input.size(1)
    }

    if (input.isContiguous()) {
      output = input.view(inferedSizes)
    } else {
      output = input.contiguous().view(inferedSizes)
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    if (gradOutput.isContiguous()) {
      gradInput = gradOutput.view(input.size())
    } else {
      gradInput = gradOutput.contiguous().view(input.size())
    }
    gradInput
  }

  override def equals(obj: Any): Boolean = {

    if (!super.equals(obj)) {
      return false
    }

    if (!obj.isInstanceOf[InferReshape[T]]) {
      return false
    }
    val other = obj.asInstanceOf[InferReshape[T]]
    if (this.eq(other)) {
      return true
    }

    var i = 0
    while (i < inferedSizes.length) {
      if (inferedSizes(i) != other.inferedSizes(i)) {
        return false
      }
      i += 1
    }
    batchMode == other.batchMode
  }

  override def hashCode(): Int = {
    val seed = 37
    var hash = super.hashCode()
    var i = 0
    while (i < inferedSizes.length) {
      hash = hash * seed + inferedSizes(i).hashCode()
      i += 1
    }
    hash = hash * seed + batchMode.hashCode()

    hash
  }

  override def toString(): String = {
    s"nn.Reshape(${
      size.mkString("x")
    })"
  }
}

object InferReshape {
  def apply[@specialized(Float, Double) T: ClassTag](size: Array[Int], batchMode: Boolean = false)
    (implicit ev: TensorNumeric[T]): InferReshape[T] =
    new InferReshape(size, batchMode)
}
