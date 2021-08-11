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
import com.intel.analytics.bigdl.utils.Shape

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/**
 * Reshape the input tensor with automatic size inference support.
 * Positive numbers in the `size` argument are used to reshape the input to the
 * corresponding dimension size.
 * There are also two special values allowed in `size`:
 *    a. `0` means keep the corresponding dimension size of the input unchanged.
 *       i.e., if the 1st dimension size of the input is 2,
 *       the 1st dimension size of output will be set as 2 as well.
 *    b. `-1` means infer this dimension size from other dimensions.
 *       This dimension size is calculated by keeping the amount of output elements
 *       consistent with the input.
 *       Only one `-1` is allowable in `size`.
 *
 * For example,
 *    Input tensor with size: (4, 5, 6, 7)
 *    -> InferReshape(Array(4, 0, 3, -1))
 *    Output tensor with size: (4, 5, 3, 14)
 * The 1st and 3rd dim are set to given sizes, keep the 2nd dim unchanged,
 * and inferred the last dim as 14.
 * @param size      the target tensor size
 * @param batchMode whether in batch mode
 * @tparam T Numeric type ([[Float]] and [[Double]] are allowed)
 */
class InferReshape[T: ClassTag](
  size: Array[Int], var batchMode: Boolean = false)(
  implicit ev: TensorNumeric[T]) extends TensorModule[T] {
  private var inferedSizes: Array[Int] = _
  private var startIndex = 0
  private var inferIndex = -1
  private var subTotal = 1
  private var inPlace = true

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
    require(total <= input.nElement(), "inferred size " +
      s"dim product must be <= total input #elements" +
      s"dim product($total) input(${input.nElement()})")
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
      inPlace = false
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
    s"${getPrintName}(${
      size.mkString("x")
    })"
  }

  override def clearState(): this.type = {
    if (!inPlace) {
      super.clearState()
    }
    this
  }

  override def computeOutputShape(inputShape: Shape): Shape = {
    val inputSize = inputShape.toSingle().toArray
    val outputSize = new ArrayBuffer[Int]()
    inferedSizes.foreach(outputSize.append(_))

    var total = subTotal
    var i = 0
    while (i < size.length) {
      if (size(i) == 0) { // use the same dim value as input
        outputSize(i + startIndex) = inputSize(i)
        total *= inputSize(i)
      }
      i += 1
    }
    if (inferIndex != -1) {
      outputSize(inferIndex) = inputSize.product / total
      if (batchMode) outputSize(inferIndex) = outputSize(inferIndex) / inputSize(0)
    }
    if (batchMode) outputSize(0) = inputSize(0)
    Shape(outputSize.toArray)
  }
}

object InferReshape {
  def apply[@specialized(Float, Double) T: ClassTag](size: Array[Int], batchMode: Boolean = false)
    (implicit ev: TensorNumeric[T]): InferReshape[T] =
    new InferReshape(size, batchMode)
}
