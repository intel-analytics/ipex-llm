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
package com.intel.analytics.bigdl.nn.ops

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

class All[T: ClassTag](keepDim : Boolean = false, startFromZero : Boolean = false)
  (implicit ev: TensorNumeric[T]) extends Operation[Table,
  Tensor[Boolean], T] {

  output = Tensor[Boolean]()

  private var buffer = Tensor[Boolean]()

  override def updateOutput(input: Table): Tensor[Boolean] = {
    val data = input[Tensor[Boolean]](1)
    val indices = input[Tensor[Int]](2)
    require(indices.nDimension() == 1 || indices.isScalar, "indices must be 1D tensor or scala")
    output.resizeAs(data)
    buffer.resizeAs(data).copy(data)
    val reduceDims = new ArrayBuffer[Int]()
    val size = output.size()
    if (indices.isScalar) {
      val dim = if (indices.value() < 0) {
        data.nDimension() + indices.value() + 1
      } else if (startFromZero) {
        indices.value() + 1
      } else {
        indices.value()
      }

      if (size(dim - 1) != 1) {
        size(dim - 1) = 1
        reduceDims += dim
        output.resize(size)
        buffer.reduce(dim, output, (a, b) => a && b)
        buffer.resizeAs(output).copy(output)
      }
    } else {
      var i = 1
      while (i <= indices.size(1)) {
        val dim = if (indices.valueAt(i) < 0) {
          data.nDimension() + indices.valueAt(i) + 1
        } else if (startFromZero) {
          indices.valueAt(i) + 1
        } else {
          indices.valueAt(i)
        }
        if (size(dim - 1) != 1) {
          size(dim - 1) = 1
          reduceDims += dim
          output.resize(size)
          buffer.reduce(dim, output, (a, b) => a && b)
          buffer.resizeAs(output).copy(output)
        }
        i += 1
      }
    }

    if (!keepDim) {
      val sizeBuffer = new ArrayBuffer[Int]()
      var i = 1
      while (i <= data.nDimension()) {
        if (!reduceDims.contains(i)) sizeBuffer.append(data.size(i))
        i += 1
      }
      output.resize(sizeBuffer.toArray)
    }
    output
  }

  override def clearState(): this.type = {
    super.clearState()
    buffer.set()
    this
  }
}

object All {
  def apply[T: ClassTag](keepDim: Boolean = false, startFromZero : Boolean = false)
    (implicit ev: TensorNumeric[T]): All[T] = new All[T](keepDim, startFromZero)
}
