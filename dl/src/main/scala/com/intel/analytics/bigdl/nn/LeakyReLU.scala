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
 * It is a transfer module that applies LeakyReLU, which parameter
 * negval sets the slope of the negative part:
 * LeakyReLU is defined as:
 *  f(x) = max(0, x) + negval * min(0, x)
 *
 * @param negval sets the slope of the negative partl
 * @param inplace if it is true, doing the operation in-place without
 *                using extra state memory
 */

@SerialVersionUID(- 6870619109313859155L)
class LeakyReLU[T: ClassTag](
  negval: Double = 0.01,
  var inplace: Boolean = false)(
  implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  private val negVal = ev.fromType[Double](negval)

  if (negval < 0) {
    inplace = false
  }

  // Todo: performance should be optimized by replacing apply for contiguous input
  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    if (inplace) {
      input.apply1(x => {
        if (ev.isGreaterEq(ev.fromType[Int](0), x)) {
          negVal
        } else {
          x
        }
      })
      output.set(input)
    } else {
      output.resizeAs(input)
      output.map(input, (out, in) => {
        if (ev.isGreater(in, ev.fromType[Int](0))) {
          in
        } else {
          ev.times(in, negVal)
        }
      })
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    require(input.isSameSizeAs(gradOutput),
      "input should have the same size with gradOutput")
    if (inplace) {
      gradOutput.map(input, (grad, in) => {
        if (ev.isGreaterEq(ev.fromType[Int](0), in)) {
          negVal
        } else {
          grad
        }
      })
    } else {
      gradInput.resizeAs(input)
      val func = new TensorFunc6[T] {
        override def apply (data1: Array[T], offset1: Int, data2: Array[T],
          offset2: Int, data3: Array[T], offset3: Int): Unit = {
          data1(offset1) = if (ev.isGreater(data3(offset3), ev.fromType[Int](0))) {
            data2(offset2)
          } else {
            ev.times(negVal, data2(offset2))
          }
        }
      }
      DenseTensorApply.apply3[T](gradInput, gradOutput, input, func)
    }
    gradInput
  }

  override def toString(): String = {
    s"nn.LeakyReLU"
  }
}

object LeakyReLU {
  def apply[@specialized(Float, Double) T: ClassTag](
      negval: Double = 0.01,
      inplace: Boolean = false)(implicit ev: TensorNumeric[T]) : LeakyReLU[T] = {
    new LeakyReLU[T](negval, inplace)
  }
}
