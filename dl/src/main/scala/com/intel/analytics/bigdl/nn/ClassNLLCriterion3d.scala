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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import scala.reflect.ClassTag

/**
 * This class is intended to support inputs with 3 or more dimensions.
 * Apply Negative Log Likelihood Criterion to every temporal slice of an input.
 * @param weights
 * @param sizeAverage
 * @param timeDim
 */

class ClassNLLCriterion3d[T : ClassTag](
  weights: Tensor[T] = null,
  sizeAverage: Boolean = true,
  timeDim: Int = 2)
(implicit ev: TensorNumeric[T]) extends ClassNLLCriterion[T] {

  private var fInput: Tensor[T] = _
  private var fTarget: Tensor[T] = _
  private var inputSize: Array[Int] = _
  private var targetSize: Array[Int] = _

  private def combine(src: Array[Int], target: Array[Int]): Unit = {
    require(src.length == target.length + 1,
      "TimeDistributed: combine method requires src.length == target.length + 1" +
        s" Current src.length = ${src.length}" +
        s" Current target.length = ${target.length}")

    target(0) = src(0) * src(1)
    var j = 1
    while (j < target.length) {
      target(j) = src(j + 1)
      j += 1
    }
  }

  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    require(input.dim() >= 3,
      "input should be at least a 3D Tensor, e.g.[batch, time, inputDim]. "
        + s"Current input.dim = ${input.dim}")

//    fInput = input.transpose(batchDim, timeDim).contiguous
//    fTarget = target.transpose(batchDim, timeDim).contiguous
//    times = input.size(timeDim)

    if (inputSize == null) {
      inputSize = new Array[Int](input.size.length - 1)
    }
    if (targetSize == null) {
      targetSize = new Array[Int](target.size.length - 1)
    }

    combine(input.size, inputSize)
    combine(target.size, targetSize)
    fInput = input.reshape(inputSize)
    fTarget = target.reshape(targetSize)
    output = super.updateOutput(fInput, fTarget)
    output
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    require(input.dim() >= 3,
      "input should be at least a 3D Tensor, e.g.[batch, time, inputDim]. "
        + s"Current input.dim = ${input.dim}")

    if (inputSize == null) {
      inputSize = new Array[Int](input.size.length - 1)
    }
    if (targetSize == null) {
      targetSize = new Array[Int](target.size.length - 1)
    }

    combine(input.size, inputSize)
    combine(target.size, targetSize)
    fInput = input.reshape(inputSize)
    fTarget = target.reshape(targetSize)
    val _gradInput = super.updateGradInput(fInput, fTarget).toTensor[T]

    require(_gradInput.nElement() == input.nElement(),
      "updateGradInput: layer gradInput size should be matchable" +
        s"to input size, current layer gradInput size dimensions is ${_gradInput.nElement()}," +
        s"input size dimensions is ${input.nElement()}")

    gradInput = _gradInput.reshape(input.size)
    gradInput
  }

  object ClassNLLCriterion3d {
    def apply[@specialized(Float, Double) T: ClassTag](
        weights: Tensor[T] = null,
        sizeAverage: Boolean = true,
        timeDim: Int = 2)(implicit ev: TensorNumeric[T]) : ClassNLLCriterion3d[T] = {
      new ClassNLLCriterion3d[T](weights, sizeAverage, timeDim)
    }
  }
}

