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
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

/**
 * Takes a table of Tensors and outputs the max of all of them.
 */

@SerialVersionUID(8594258233874356842L)
class CMaxTable[T: ClassTag](implicit ev: TensorNumeric[T])
  extends AbstractModule[Table, Tensor[T], T]{

  @transient
  private var maxIdx: Tensor[T] = null
  @transient
  private var mask: Tensor[T] = null
  @transient
  private var maskResult: Tensor[T] = null

  override def updateOutput(input: Table): Tensor[T] = {
    if (null == maxIdx) maxIdx = Tensor[T]()
    if (null == mask) mask = Tensor[T]()
    if (null == maskResult) maskResult = Tensor[T]()

    val res1 = input[Tensor[T]](1)
    output.resizeAs(res1).copy(res1)
    maxIdx.resizeAs(res1).fill(ev.fromType(1))

    var i = 2
    while (i <= input.length()) {
      mask.resize(res1.size())
      mask.gt(input(i), output)
      maxIdx.maskedFill(mask, ev.fromType(i))

      if (ev.isGreater(mask.sum(), ev.fromType(0))) {
        output.maskedCopy(mask, input[Tensor[T]](i).maskedSelect(mask, maskResult))
      }
      i += 1
    }

    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]): Table = {
    var i = 1
    while (i <= input.length()) {
      if (!gradInput.contains(i)) gradInput.insert(i, Tensor[T]())
      gradInput[Tensor[T]](i).resizeAs(input(i)).zero()

      mask.resize(maxIdx.size())
      mask.eq(maxIdx, ev.fromType(i))

      if (ev.isGreater(mask.sum(), ev.fromType(0))) {
        gradInput[Tensor[T]](i).maskedCopy(mask, gradOutput.maskedSelect(mask, maskResult))
      }
      i += 1
    }
    gradInput
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[CMaxTable[T]]

  override def equals(other: Any): Boolean = other match {
    case that: CMaxTable[T] =>
      super.equals(that) &&
        (that canEqual this)
    case _ => false
  }

  override def hashCode(): Int = {
    def getHashCode(a: Any): Int = if (a == null) 0 else a.hashCode()
    val state = Seq(super.hashCode())
    state.map(getHashCode).foldLeft(0)((a, b) => 31 * a + b)
  }
}

object CMaxTable {
  def apply[@specialized(Float, Double) T: ClassTag]()
      (implicit ev: TensorNumeric[T]) : CMaxTable[T] = {
    new CMaxTable[T]()
  }
}
