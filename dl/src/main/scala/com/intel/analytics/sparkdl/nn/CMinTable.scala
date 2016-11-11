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

/**
 * Takes a table of Tensors and outputs the min of all of them.
 */
class CMinTable[T: ClassTag](implicit ev: TensorNumeric[T])
  extends Module[Table, Tensor[T], T]{

  @transient
  private var minIdx: Tensor[T] = null
  @transient
  private var mask: Tensor[T] = null

  override def updateOutput(input: Table): Tensor[T] = {
    if (null == minIdx) minIdx = Tensor[T]()
    if (null == mask) mask = Tensor[T]()

    val res1 = input[Tensor[T]](1)
    output.resizeAs(res1).copy(res1)
    minIdx.resizeAs(res1).fill(ev.fromType(1))

    var i = 2
    while (i <= input.length()) {
      mask.resize(res1.size())
      mask.lt(input(i), output)
      minIdx.maskedFill(mask, ev.fromType(i))

      val maskResult = Tensor[T]()
      output.maskedCopy(mask, input[Tensor[T]](i).maskedSelect(mask, maskResult))
      i += 1
    }
    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]): Table = {
    var i = 1
    while (i <= input.length()) {
      if (!gradInput.contains(i)) gradInput.insert(i, Tensor[T]())
      gradInput[Tensor[T]](i).resizeAs(input(i)).zero()

      mask.resize(minIdx.size())
      mask.eq(minIdx, ev.fromType(i))

      val maskResult = Tensor[T]()
      gradInput.apply[Tensor[T]](i).maskedCopy(mask, gradOutput.maskedSelect(mask, maskResult))

      i += 1
    }

    gradInput
  }

  override def toString() : String = {
    "nn.CMinTable"
  }

}
