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
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/**
 * Expand tensor to configured size
 * @param targetSizes target tensor sizes, dim whose size is -1 will be ignored
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class ExpandSize[T: ClassTag](targetSizes: Array[Int])
   (implicit ev: TensorNumeric[T]) extends AbstractModule[Tensor[T], Tensor[T], T] {

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(targetSizes.length == input.dim(),
      s"the number of dimensions provided must equal ${input.dim()}")
    val tensorDim = input.dim()
    val tensorStride = input.stride()
    val tensorSize = input.size()

    var i = 0
    while (i < tensorDim) {
      if (targetSizes(i) != -1) {
        if (tensorSize(i) == 1) {
          tensorSize(i) = targetSizes(i)
          tensorStride(i) = 0
        } else if (tensorSize(i) != targetSizes(i)) {
          throw new UnsupportedOperationException(
            "incorrect size: only supporting singleton expansion (size=1)")
        }
      }
      i += 1
    }

    output.set(input.storage(), input.storageOffset(), tensorSize, tensorStride)
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    val tensorDim = input.dim()
    val tensorSize = input.size()

    gradInput = Tensor[T](tensorSize)
    val expandDim = new ArrayBuffer[Int]()
    var i = 0
    while (i < tensorDim) {
      if (targetSizes(i) != -1) {
        if (tensorSize(i) == 1 && targetSizes(i) != 1) {
          expandDim.append(i + 1)
        }
      }
      i += 1
    }

    i = expandDim.size - 1
    val sizes = gradOutput.size()
    var _gradOutput = gradOutput
    while (i >= 0) {
      var start = 1
      sizes(expandDim(i) - 1) = 1
      val _gradInput = Tensor[T](sizes)
      while (start <= gradOutput.size(expandDim(i))) {
        val x = _gradOutput.narrow(expandDim(i), start, 1)
        _gradInput.add(x)
        start += 1
      }
      _gradOutput = _gradInput
      i -= 1
    }
    gradInput = _gradOutput
    gradInput
  }

  override def toString: String = s"ExpandSize"
}

object ExpandSize {
  def apply[@specialized(Float, Double) T: ClassTag](targetSizes: Array[Int])
     (implicit ev: TensorNumeric[T]) : ExpandSize[T] = {
    new ExpandSize[T](targetSizes)
  }
}
