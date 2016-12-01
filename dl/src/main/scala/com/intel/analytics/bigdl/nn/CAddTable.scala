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

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect._

class CAddTable[@specialized(Float, Double) T: ClassTag](val inplace: Boolean = false)(
  implicit ev: TensorNumeric[T]) extends Module[Table, Tensor[T], T] {

  gradInput = T()

  override def updateOutput(input: Table): Tensor[T] = {
    if (inplace) {
      output.set(input[Tensor[T]](1))
      //output = input[Tensor[T]](1)
    } else {
      output.resizeAs(input[Tensor[T]](1)).copy(input[Tensor[T]](1))
     /* val input1 = input[Tensor[T]](1)
      if (null == output) {
        output = input1.clone()
      } else {
        output.resizeAs(input1).copy(input1)
      }*/
    }

    var i = 2
    while (i <= input.length()) {
      output.add(input[Tensor[T]](i))
      i += 1
    }

    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]) : Table = {
    var i = 1
    while (i <= input.length()) {
      if (i > gradInput.length) gradInput.insert(i, Tensor[T]().resizeAs(input(1)))
      if (inplace) {
        gradInput[Tensor[T]](i).set(gradOutput) // = gradOutput
      } else {
//        if (gradInput.contains(i)) {
//          gradInput[Tensor[T]](i).resizeAs(gradOutput).copy(gradOutput)
        gradInput[Tensor[T]](i).resizeAs(gradOutput).copy(gradOutput)
      }
      i += 1
    }
    i = input.length + 1
    while (i <= gradInput.length) {
      gradInput.remove(i)
    }
    gradInput
  }

  override def toString() : String = {
    "nn.CAddTable"
  }
}

object Test {
  def main(args: Array[String]): Unit = {
    val a = T()
    a.update(1, Tensor[Double]())
  }
}
