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

import com.intel.analytics.bigdl.nn.abstractnn.{IdentityOutputShape, TensorModule}
import com.intel.analytics.bigdl.tensor.{DenseTensorApply, Tensor, TensorFunc6}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * Apply Segment-wise linear approximation of sigmoid.
 * Faster than sigmoid
 *           ⎧  0, if x < -2.5
 *    f(x) = ⎨  1, if x > 2.5
 *           ⎩  0.2 * x + 0.5, otherwise
 */
class HardSigmoid[T: ClassTag]
(implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  val minValue = ev.fromType[Double](-2.5)
  val maxValue = ev.fromType[Double](2.5)

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output.resizeAs(input)
    output.map(input, (out, in) => {
      if (ev.isGreater(in, maxValue)) {
        ev.fromType[Int](1)
      } else if (ev.isGreater(minValue, in)) {
        ev.fromType[Int](0)
      } else {
        ev.fromType[Double](0.2 * ev.toType[Double](in) + 0.5)
      }
    })
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    require(input.isSameSizeAs(gradOutput),
      "Input should have the same size as gradOutput" +
        s"input size(${input.dim()}) gradOutput size(${gradOutput.dim()})")
    gradInput.resizeAs(input)
    val func = new TensorFunc6[T] {
      override def apply(data1: Array[T], offset1: Int, data2: Array[T],
                         offset2: Int, data3: Array[T], offset3: Int): Unit = {
        if (ev.isGreater(data3(offset3), maxValue)
          || ev.isGreater(minValue, data3(offset3))) {
          data1(offset1) = ev.fromType[Double](0)
        } else {
          data1(offset1) = ev.times(data2(offset2), ev.fromType[Double](0.2))
        }
      }
    }
    DenseTensorApply.apply3[T](gradInput, gradOutput, input, func)
    gradInput
  }
}

object HardSigmoid {
  def apply[T : ClassTag]()(implicit ev: TensorNumeric[T]): HardSigmoid[T] = new HardSigmoid[T]()
}
