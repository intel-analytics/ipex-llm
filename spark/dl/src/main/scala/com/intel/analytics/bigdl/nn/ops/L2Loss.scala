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
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor._

import scala.reflect.ClassTag

class L2Loss[T: ClassTag]()
  (implicit ev: TensorNumeric[T]) extends Operation[Tensor[_], Tensor[_], T] {
  var buffer: Tensor[_] = Tensor[Float]()

  override def updateOutput(input: Tensor[_]): Tensor[_] = {
    input.getType() match {
      case FloatType =>
        if (output.getType() != FloatType) {
          output = Activity.allocate[Tensor[Float], Float]()
        }
        if (buffer.getType() != FloatType) {
          buffer = Activity.allocate[Tensor[Float], Float]()
        }
        buffer.resizeAs(input)
        output.resize(1)
        output.asInstanceOf[Tensor[Float]].setValue(1,
          buffer.asInstanceOf[Tensor[Float]].applyFun[Float](
            input.asInstanceOf[Tensor[Float]], x => x * x).sum() / 2)
      case DoubleType =>
        if (output.getType() != DoubleType) {
          output = Activity.allocate[Tensor[Double], Double]()
        }
        val a = buffer.getType()
        if (buffer.getType() != DoubleType) {
          buffer = Activity.allocate[Tensor[Double], Double]()
        }
        buffer.resizeAs(input)
        output.resize(1)
        output.asInstanceOf[Tensor[Double]].setValue(1,
          buffer.asInstanceOf[Tensor[Double]].applyFun[Double](
            input.asInstanceOf[Tensor[Double]], x => x * x).sum() / 2)
      case _ => throw new RuntimeException("Unsupported tensor type")
    }

    output
  }
}

object L2Loss {
  def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]):
  Operation[Activity, Activity, T]
  = ModuleToOperation[T](new L2Loss())
}
