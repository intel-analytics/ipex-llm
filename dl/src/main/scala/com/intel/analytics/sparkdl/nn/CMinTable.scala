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

import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.utils.Table

import scala.reflect.ClassTag

class CMinTable[T: ClassTag](implicit ev: TensorNumeric[T])
  extends Module[Table, Tensor[T], T]{

  @transient
  var minIdx: Tensor[T] = null

  override def updateOutput(input: Table): Tensor[T] = {
    val res1 = input[Tensor[T]](1)
    val res2 = input[Tensor[T]](2)

    if (null == minIdx) minIdx = Tensor[T]()
    output.resizeAs(res1).copy(res1)
    minIdx.resizeAs(res1).fill(ev.fromType(1))

    var i = 2
    while (i <= input.length()) {
      val mask = Tensor[T].resize(res1.size())
      mask.lt(input(i), output)
      minIdx.maskedFill(mask, ev.fromType(i))

      output.maskedCopy(mask, Tensor[T].maskedSelect(mask, input(i)))
      i += 1
    }
    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]): Table = {
    var i = 1
    while (i <= input.length()) {
      gradInput.insert(i, Tensor[T]())
      gradInput[Tensor[T]](i).resizeAs(input(i)).zero()

      val mask = Tensor[T].resize(minIdx.size())
      mask.eq(minIdx, ev.fromType(i))

      gradInput.apply[Tensor[T]](i).maskedCopy(mask, Tensor[T].maskedSelect(mask, gradOutput))

      i += 1
    }

    gradInput
  }

  override def toString() : String = {
    "nn.CMinTable"
  }

}
