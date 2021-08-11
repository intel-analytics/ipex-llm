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
 * Takes a table of Tensors and outputs the multiplication of all of them.
 */

@SerialVersionUID(8888147326550637025L)
class CMulTable[T: ClassTag]()(
  implicit ev: TensorNumeric[T]) extends AbstractModule[Table, Tensor[T], T]{

  private var scalarIndexes : Array[Int] = _

  override def updateOutput(input: Table): Tensor[T] = {
    var scalar = ev.one
    var hasTensor = false
    var hasScalar = false
    var initTensor = false
    var i = 1
    while (i <= input.length()) {
      val curTensor = input[Tensor[T]](i)
      if (curTensor.isScalar) {
        scalar = ev.times(scalar, curTensor.value())
        hasScalar = true
      } else if (curTensor.isTensor) {
        if (initTensor) {
          output.cmul(curTensor)
        } else {
          output.resizeAs(curTensor).copy(curTensor)
          initTensor = true
        }
        hasTensor = true
      }
      i += 1
    }

    if (hasTensor && hasScalar) {
      output.mul(scalar)
    } else if (hasScalar) {
      output.resizeAs(input[Tensor[T]](1)).setValue(scalar)
    }

    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]) : Table = {
    var i = 1
    while (i <= input.length()) {
      if (!gradInput.contains(i)) gradInput.insert(i, Tensor[T]())
      gradInput[Tensor[T]](i).resizeAs(gradOutput).copy(gradOutput)
      var j = 1
      while (j <= input.length()) {
        if (i != j) {
          if (input[Tensor[T]](j).isScalar) {
            gradInput[Tensor[T]](i).mul(input[Tensor[T]](j).value())
          } else {
            gradInput[Tensor[T]](i).cmul(input(j))
          }
        }
        j += 1
      }
      if (input[Tensor[T]](i).isScalar) {
        val sum = gradInput[Tensor[T]](i).sum()
        gradInput(i) = gradInput[Tensor[T]](i).resizeAs(input(i)).setValue(sum)
      }
      i += 1
    }
    gradInput
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[CMulTable[T]]

  override def equals(other: Any): Boolean = other match {
    case that: CMulTable[T] =>
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

object CMulTable {
  def apply[@specialized(Float, Double) T: ClassTag]()
      (implicit ev: TensorNumeric[T]) : CMulTable[T] = {
    new CMulTable[T]()
  }
}
