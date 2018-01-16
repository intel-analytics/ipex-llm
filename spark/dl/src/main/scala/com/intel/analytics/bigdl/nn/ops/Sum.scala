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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table
import com.intel.analytics.bigdl.nn.{Sum => SumLayer}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

class Sum[T: ClassTag, D: ClassTag](val keepDims: Boolean, val startFromZero: Boolean = false)
  (implicit ev: TensorNumeric[T], ev2: TensorNumeric[D])
  extends Operation[Table, Tensor[D], T] {

  private val sum: SumLayer[D] = SumLayer[D](squeeze = !keepDims)

  output = Tensor[D]()

  override def updateOutput(input: Table): Tensor[D] = {
    val data = input[Tensor[D]](1)
    val dims = input[Tensor[Int]](2)

    output.resizeAs(data).copy(data)

    val sumDims = if (dims.isEmpty) {
      return output
    } else if (dims.isScalar) {
      Array(if (startFromZero) dims.value() + 1 else dims.value())
    } else {
      require(dims.nDimension() == 1, s"Only accept 1D as dims, but now is ${dims.nDimension()}")
      val buffer = new ArrayBuffer[Int]()
      dims.apply1(a => {
        buffer.append(if (startFromZero) a + 1 else a)
        a
      })
      buffer.toArray.sortWith(_ > _)
    }

    var i = 0
    while(i < sumDims.length) {
      sum.changeSumDims(sumDims(i))
      val tmp = sum.updateOutput(output)
      output.resizeAs(tmp).copy(tmp)
      i += 1
    }

    output
  }

  override def getClassTagNumerics() : (Array[ClassTag[_]], Array[TensorNumeric[_]]) = {
    (Array[ClassTag[_]](scala.reflect.classTag[T], scala.reflect.classTag[D]),
      Array[TensorNumeric[_]](ev, ev2))
  }
}

object Sum {
  def apply[T: ClassTag, D: ClassTag](keepDims: Boolean = false, startFromZero: Boolean = false)
    (implicit ev: TensorNumeric[T], ev2: TensorNumeric[D]): Sum[T, D] =
    new Sum(keepDims, startFromZero)
}
