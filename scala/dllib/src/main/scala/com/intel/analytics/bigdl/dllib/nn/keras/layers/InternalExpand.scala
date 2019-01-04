/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.pipeline.api.keras.layers.internal

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

/**
 * Expand tensor to configured size

 * @param tgtSizes target tensor sizes, dim whose size is -1 will be ignored
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
private[zoo] class InternalExpand[T: ClassTag](tgtSizes: Array[Int])
  (implicit ev: TensorNumeric[T]) extends AbstractModule[Tensor[T], Tensor[T], T] {

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(tgtSizes.length == input.dim(),
      s"the number of dimensions provided must equal ${input.dim()}")
    val tensorDim = input.dim()
    val tensorStride = input.stride()
    val tensorSize = input.size()

    // check if need batch dim
    var i = 0
    while (i < tensorDim) {
      if (tgtSizes(i) != -1) {
        if (tensorSize(i) == 1) {
          tensorSize(i) = tgtSizes(i)
          tensorStride(i) = 0
        } else if (tensorSize(i) != tgtSizes(i)) {
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

    var i = 0
    while (i < tensorDim) {
      if (tgtSizes(i) != -1) {
        if (tensorSize(i) == 1) {
          gradOutput.split(i + 1).foreach(gradInput.add(_))
        }
      }
      i += 1
    }
    gradInput
  }

  override def toString: String = s"InternalExpand()"
}

private[zoo] object InternalExpand {
  def apply[@specialized(Float, Double) T: ClassTag](tgtSizes: Array[Int])
    (implicit ev: TensorNumeric[T]) : InternalExpand[T] = {
    new InternalExpand[T](tgtSizes)
  }
}
