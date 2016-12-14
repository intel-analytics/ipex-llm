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

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

class CAddTable[@specialized(Float, Double) T: ClassTag](val inplace: Boolean = false)(
  implicit ev: TensorNumeric[T]) extends AbstractModule[Table, Tensor[T], T] {

  override def updateOutput(input: Table): Tensor[T] = {
    if (inplace) {
      output = input[Tensor[T]](1)
    } else {
      val input1 = input[Tensor[T]](1)
      if (null == output) {
        output = input1.clone()
      } else {
        output.resizeAs(input1).copy(input1)
      }
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
      if (inplace) {
        gradInput(i) = gradOutput
      } else {
        if (gradInput.contains(i)) {
          gradInput[Tensor[T]](i).resizeAs(gradOutput).copy(gradOutput)
        } else {
          gradInput.insert(i, gradOutput.clone())
        }
      }
      i += 1
    }

    gradInput
  }

  override def toString() : String = {
    "nn.CAddTable"
  }
}

object CAddTable {
  def apply[@specialized(Float, Double) T: ClassTag](
      inplace: Boolean = false)(implicit ev: TensorNumeric[T]) : CAddTable[T] = {
    new CAddTable[T](inplace)
  }
}
