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
import com.intel.analytics.bigdl.tensor.{DenseTensorApply, Tensor, TensorFunc6}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * This is a transfer layer which applies the hard shrinkage function
 * element-wise to the input Tensor. The parameter lambda is set to 0.5
 * by default
 *        ⎧ x, if x >  lambda
 * f(x) = ⎨ x, if x < -lambda
 *        ⎩ 0, otherwise
 *
 * @param lambda: a threshold value whose default value is 0.5
 */

@SerialVersionUID( 3551967457354343585L)
class HardShrink[T: ClassTag](private val lambda: Double = 0.5)
  (implicit ev: TensorNumeric[T])
  extends TensorModule[T] {
  private val lam = ev.fromType[Double](lambda)
  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output.resizeAs(input)
    output.map(input, (out, in) => {
      if (ev.isGreater(in, lam) || ev.isGreater(ev.negative(lam), in)) {
        in
      } else {
        ev.fromType[Int](0)
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
        if (ev.isGreater(data3(offset3), lam)
          || ev.isGreater(ev.negative(lam), data3(offset3))) {
          data1(offset1) = data2(offset2)
        } else {
          data1(offset1) = ev.fromType[Double](0)
        }
      }
    }
    DenseTensorApply.apply3[T](gradInput, gradOutput, input, func)
    gradInput
  }
}

object HardShrink {
  def apply[@specialized(Float, Double) T: ClassTag](
      lambda: Double = 0.5)(implicit ev: TensorNumeric[T]) : HardShrink[T] = {
    new HardShrink[T](lambda)
  }
}
