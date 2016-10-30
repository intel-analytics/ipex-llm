/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
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
package com.intel.analytics.sparkdl.nn

import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.tensor.{DenseTensorApply, Tensor, TensorFunc4, TensorFunc6}

import scala.reflect.ClassTag

class Square[T: ClassTag](
  implicit ev: TensorNumeric[T]) extends TensorModule[T] {
  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output.resizeAs(input)

    if (input.dim() == 1 || !input.isContiguous() || !output.isContiguous()) {
      val func = new TensorFunc4[T] {
        override def apply(data1: Array[T], offset1: Int, data2: Array[T], offset2: Int): Unit = {
          data1(offset1) = ev.times(data2(offset2), data2(offset2))
        }
      }
      DenseTensorApply.apply2[T](output, input, func)
    } else {
      val input_data = input.storage().array()
      val output_data = output.storage().array()

      var i = 0
      while (i < output_data.length) {
        output_data(i) = ev.times(input_data(i), input_data(i))
        i += 1
      }
    }

    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    require(input.isSameSizeAs(gradOutput),
      "input and gradOutput need to have the same shape")
    gradInput.resizeAs(input)

    if (input.dim() == 1 || !input.isContiguous() || !gradOutput.isContiguous()
      || !gradInput.isContiguous()) {
      val func = new TensorFunc6[T] {
        override def apply(data1: Array[T], offset1: Int, data2: Array[T], offset2: Int,
          data3: Array[T], offset3: Int): Unit = {
          data1(offset1) = ev.times(
            ev.times(data2(offset2), data3(offset3)), ev.fromType[Int](2))
        }
      }
      DenseTensorApply.apply3[T](gradInput, gradOutput, input, func)
    } else {
      val gradOutput_data = gradOutput.storage().array()
      val gradInput_data = gradInput.storage().array()
      val input_data = input.storage().array()

      var i = 0
      while (i < gradInput_data.length) {
        gradInput_data(i) = ev.times(
          ev.times(gradOutput_data(i), input_data(i)), ev.fromType[Int](2))
        i += 1
      }
    }

    gradInput
  }

  override def toString(): String = {
    s"nn.Square"
  }
}
