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

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor._

import scala.reflect.ClassTag

class L2Loss[T: ClassTag]()
  (implicit ev: TensorNumeric[T]) extends Operation[Tensor[T], T] {
  val buffer: Tensor[T] = Tensor[T]()

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output.resize(1)
    buffer.resizeAs(input)
    input.getType() match {
      case FloatType =>
        output.asInstanceOf[Tensor[Float]].setValue(1,
          buffer.asInstanceOf[Tensor[Float]].applyFun[Float](
            input.asInstanceOf[Tensor[Float]], x => x * x).sum() / 2)
      case DoubleType =>
        output.asInstanceOf[Tensor[Double]].setValue(1,
          buffer.asInstanceOf[Tensor[Double]].applyFun[Double](
            input.asInstanceOf[Tensor[Double]], x => x * x).sum() / 2)
      case _ => throw new RuntimeException("Unsupported tensor type")
    }

    output
  }
}

object L2Loss {
  def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]): Operation[Tensor[T], T]
  = ModuleToOperation[Tensor[T], T](new L2Loss())
}