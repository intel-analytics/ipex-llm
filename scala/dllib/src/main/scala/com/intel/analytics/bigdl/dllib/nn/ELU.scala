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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, TensorModule}
import com.intel.analytics.bigdl.tensor.{DenseTensorApply, Tensor, TensorFunc6}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * Djork-Arné Clevert, Thomas Unterthiner, Sepp Hochreiter
 * Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
 *   [http://arxiv.org/pdf/1511.07289.pdf]
 */

@SerialVersionUID( - 3525781855978085005L)
class ELU[T: ClassTag, D: ClassTag](
  val alpha: Double = 1.0,
  val inplace: Boolean = false)(
  implicit ev: TensorNumeric[T], ev2: TensorNumeric[D])
  extends AbstractModule[Tensor[D], Tensor[D], T]  {

  output = Tensor[D]()
  gradInput = Tensor[D]()

  val _alpha = ev2.fromType[Double](alpha)

  // Todo: Improve the performance of contiguous tensor
  override def updateOutput(input: Tensor[D]): Tensor[D] = {
    if (inplace) {
      input.apply1(in => {
        if (ev2.isGreaterEq(ev2.fromType[Double](0), in)) {
          ev2.times(ev2.minus(ev2.exp(in), ev2.fromType[Double](1)), _alpha)
        } else {
          in
        }
      })
      output.set(input)
    } else {
      output.resizeAs(input)
      output.map(input, (out, in) => {
        if (ev2.isGreaterEq(ev2.fromType[Int](0), in)) {
          ev2.times(ev2.minus(ev2.exp(in), ev2.fromType[Double](1)), _alpha)
        } else {
          in
        }
      })
    }
    output
  }

  override def updateGradInput(input: Tensor[D], gradOutput: Tensor[D]): Tensor[D] = {
    require(input.isSameSizeAs(gradOutput),
      "input should have the same size with gradOutput" +
        s"input (${input.dim()}) gradOutput (${gradOutput.dim()}")
    if (inplace) {
      gradOutput.map(output, (grad, out) => {
        if (ev2.isGreaterEq(ev2.fromType[Int](0), out)) {
          ev2.times(ev2.plus(out, _alpha), grad)
        } else {
          grad
        }
      })
      gradInput.set(gradOutput)
    } else {
      gradInput.resizeAs(input)
      val func = new TensorFunc6[D] {
        override def apply (data1: Array[D], offset1: Int, data2: Array[D],
          offset2: Int, data3: Array[D], offset3: Int): Unit = {
          data1(offset1) = if (ev2.isGreater(data3(offset3), ev2.fromType[Int](0))) {
            data2(offset2)
          } else {
            ev2.times(ev2.plus(data3(offset3), _alpha), data2(offset2))
          }
        }
      }
      DenseTensorApply.apply3[D](gradInput, gradOutput, output, func)
    }
    gradInput
  }

  override def clearState(): this.type = {
    if (!inplace) {
      super.clearState()
    }
    this
  }

  override def getClassTagNumerics() : (Array[ClassTag[_]], Array[TensorNumeric[_]]) = {
    (Array[ClassTag[_]](scala.reflect.classTag[T], scala.reflect.classTag[D]),
      Array[TensorNumeric[_]](ev, ev2))
  }
}

object ELU {
  def apply[@specialized(Float, Double) T: ClassTag, D: ClassTag](
      alpha: Double = 1.0,
      inplace: Boolean = false)
      (implicit ev: TensorNumeric[T], ev2: TensorNumeric[D]) : ELU[T, D] = {
    new ELU[T, D](alpha, inplace)
  }
}
