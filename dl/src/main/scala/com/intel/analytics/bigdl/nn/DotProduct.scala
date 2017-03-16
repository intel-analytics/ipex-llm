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

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

/**
 * This is a simple table layer which takes a table of two tensors as input
 * and calculate the dot product between them as outputs
 */

@SerialVersionUID(2455897411271580599L)
class DotProduct[T: ClassTag] (implicit ev: TensorNumeric[T])
  extends AbstractModule[Table, Tensor[T], T] {
  gradInput = T(Tensor[T](), Tensor[T]())
  @transient private var buffer: Tensor[T] = null

  override def updateOutput(input: Table): Tensor[T] = {
    var input1: Tensor[T] = input(1)
    var input2: Tensor[T] = input(2)

    if (input1.dim() == 1) {
      input1 = input1.view(1, input1.size(1))
      input2 = input2.view(1, input2.size(1))
    }
    if (buffer == null) {
      buffer = Tensor[T]()
    }
    buffer.resizeAs(input1).cmul(input1, input2)
    output.sum(buffer, 2)
    output.resize(input1.size(1))
    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]): Table = {
    var input1: Tensor[T] = input(1)
    var input2: Tensor[T] = input(2)
    var notBatch = false

    if (gradInput.length() != 2) {
      if (!gradInput.contains(1)) {
        gradInput.update(1, Tensor[T]())
      }
      if (!gradInput.contains(2)) {
        gradInput.update(2, Tensor[T]())
      }
    }

    if (input1.dim() == 1) {
      input1 = input1.view(1, input1.size(1))
      input2 = input2.view(1, input2.size(1))
      notBatch = true
    }

    val gw1: Tensor[T] = gradInput(1)
    val gw2: Tensor[T] = gradInput(2)
    gw1.resizeAs(input1).copy(input2)
    gw2.resizeAs(input2).copy(input1)

    val go = gradOutput.view(gradOutput.size(1), 1).expandAs(input1)
    gw1.cmul(go)
    gw2.cmul(go)

    if (notBatch) {
      gradInput[Tensor[T]](1).set(gw1.select(1, 1))
      gradInput[Tensor[T]](2).set(gw2.select(1, 1))
    }

    gradInput
  }

  override def toString: String = {
    s"nn.DotProduct"
  }
}

object DotProduct {
  def apply[@specialized(Float, Double) T: ClassTag]()
      (implicit ev: TensorNumeric[T]) : DotProduct[T] = {
    new DotProduct[T]()
  }
}
