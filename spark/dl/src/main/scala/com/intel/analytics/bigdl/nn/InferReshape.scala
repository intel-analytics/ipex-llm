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
 * Reshape with the support of infered size,
 * Positive numbers are used directly, setting the corresponding dimension of the output tensor.
 * In addition, two special values are accepted:
 * 0 means "copy the respective dimension of the input".
 * i.e., if the input has 2 as its 1st dimension,
 * the output will have 2 as its 1st dimension as well
 * -1 stands for "infer this from the other dimensions"
 * this dimension is calculated to keep the overall element count the same as in the input.
 * At most one -1 can be used in a reshape operation.
 *
 * For example, (4, 5, 6, 7) -> InferReshape (4, 0, 3, -1) -> (4, 5, 3, 14)
 * with 1st and 3rd dim same as given size, with 2nd dim same as input, and the infered dim is 14
 * @param size      the target tensor size
 * @param batchMode whether in batch mode
 * @tparam T type
 */
class InferReshape[@specialized(Float, Double) T: ClassTag](
  size: Array[Int], var batchMode: Boolean = false)(
  implicit ev: TensorNumeric[T]) extends TensorModule[T] {
  private var inferedSizes: Array[Int] = _
  private var startIndex = 0
  private var inferIndex = -1
  private var subTotal = 1

  init()

  private def init(): Unit = {
    var minusOneCount = 0
    inferedSizes = if (batchMode) new Array[Int](size.length + 1) else new Array[Int](size.length)
    if (batchMode) startIndex = 1
    var i = 0
    while (i < size.length) {
      if (size(i) == -1) {
        minusOneCount += 1
        inferIndex = i + startIndex
      }
      else if (size(i) != 0) { // use the exact value in given size
        inferedSizes(i + startIndex) = size(i)
        subTotal *= size(i)
      }
      i += 1
    }
    require(minusOneCount == 1, "at most a single value of -1 may be specified")
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    var total = subTotal
    var i = 0
    while (i < size.length) {
      if (size(i) == 0) { // use the same dim value as input
        inferedSizes(i + startIndex) = input.size(i + 1)
        total *= input.size(i + 1)
      }
      i += 1
    }
    require(total <= input.nElement(), "inferred size dim product must be <= total input #elements")
    if (inferIndex != -1) {
      inferedSizes(inferIndex) = input.nElement() / total
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
    s"nn.InferReshape(${
      size.mkString("x")
    })"
  }
}

object InferReshape {
  def apply[@specialized(Float, Double) T: ClassTag](size: Array[Int], batchMode: Boolean = false)
    (implicit ev: TensorNumeric[T]): InferReshape[T] =
    new InferReshape(size, batchMode)
}
