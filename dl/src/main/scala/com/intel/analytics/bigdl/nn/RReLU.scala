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
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils.RandomGenerator._

import scala.reflect.ClassTag

@SerialVersionUID(- 9012115082607155821L)
class RReLU[T: ClassTag](
  lower: Double = 1.0/8,
  upper: Double = 1.0/3,
  inplace: Boolean = false)(
  implicit ev: TensorNumeric[T]) extends TensorModule[T]  {
  @transient
  var noise: Tensor[T] = null
  require(lower < upper && lower > 0 && upper > 0)

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    if (noise == null) {
      noise = Tensor[T]()
    }

    if (train) {
      noise.resizeAs(input)
      if (inplace) {
        val func = new TensorFunc4[T] {
          override def apply(data1: Array[T], index1: Int, data2: Array[T], index2: Int): Unit = {
            if (ev.isGreaterEq(ev.fromType[Int](0), data1(index1))) {
              val r = ev.fromType[Double](RNG.uniform(lower, upper))
              data1(index1) = ev.times(data1(index1), r)
              data2(index2) = r
            } else {
              data2(index2) = ev.fromType[Int](1)
            }
          }
        }
        DenseTensorApply.apply2[T](input, noise, func)
        output.set(input)
      } else {
        output.resizeAs(input)
        val func = new TensorFunc6[T] {
          override def apply (data1: Array[T], offset1: Int, data2: Array[T],
            offset2: Int, data3: Array[T], offset3: Int): Unit = {
            if (ev.isGreaterEq(ev.fromType[Int](0), data1(offset1))) {
              val r = ev.fromType[Double](RNG.uniform(lower, upper))
              data2(offset2) = ev.times(data1(offset1), r)
              data3(offset3) = r
            } else {
              data2(offset2) = data1(offset1)
              data3(offset3) = ev.fromType[Int](1)
            }
          }
        }
        DenseTensorApply.apply3[T](input, output, noise, func)
      }
    } else {
      val negSlope = (lower + upper) / 2
      if (inplace) {
        val func = new TensorFunc2[T] {
          override def apply(data: Array[T], index: Int): Unit = {
            if (ev.isGreaterEq(ev.fromType[Int](0), data(index))) {
              data(index) = ev.times(data(index), ev.fromType[Double](negSlope))
            }
          }
        }
        DenseTensorApply.apply1[T](input, func)
        output.set(input)
      } else {
        output.resizeAs(input)
        val func = new TensorFunc4[T] {
          override def apply(data1: Array[T], index1: Int, data2: Array[T], index2: Int): Unit = {
            val r = if (ev.isGreaterEq(ev.fromType[Int](0), data1(index1))) negSlope else 1
            data2(index2) = ev.times(ev.fromType[Double](r), data1(index1))
          }
        }
        DenseTensorApply.apply2[T](input, output, func)
      }
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    require(input.isSameSizeAs(gradOutput))
    if (noise == null) {
      noise = Tensor[T]()
    }

    if (train && upper - lower > 1E-6) {
      if (inplace) {
        gradOutput.cmul(gradOutput, noise)
        gradInput.set(gradOutput)
      } else {
        gradInput.resizeAs(input)
        gradInput.cmul(gradOutput, noise)
      }
    } else {
      val negSlope = (lower + upper) / 2
      if (inplace) {
        val func = new TensorFunc4[T] {
          override def apply(data1: Array[T], index1: Int, data2: Array[T], index2: Int): Unit = {
            if (ev.isGreaterEq(ev.fromType[Int](0), data1(index1))) {
              data1(index1) = ev.times(data1(index1), ev.fromType[Double](negSlope))
            }
          }
        }
        DenseTensorApply.apply2[T](gradOutput, input, func)
        gradInput.set(gradOutput)
      } else {
        gradInput.resizeAs(input)
        val func = new TensorFunc6[T] {
          override def apply (data1: Array[T], offset1: Int, data2: Array[T],
            offset2: Int, data3: Array[T], offset3: Int): Unit = {
            data1(offset1) = if (ev.isGreaterEq(ev.fromType[Int](0), data3(offset3))) {
              ev.times(data2(offset2), ev.fromType[Double](negSlope))
            } else {
              data2(offset2)
            }
          }
        }
        DenseTensorApply.apply3[T](gradInput, gradOutput, input, func)
      }
    }
    gradInput
  }

  override def toString: String = {
    "nn.RReLU"
  }
}

object RReLU {
  def apply[@specialized(Float, Double) T: ClassTag](
      lower: Double = 1.0/8,
      upper: Double = 1.0/3,
      inplace: Boolean = false)(implicit ev: TensorNumeric[T]) : RReLU[T] = {
    new RReLU[T](lower, upper, inplace)
  }
}
