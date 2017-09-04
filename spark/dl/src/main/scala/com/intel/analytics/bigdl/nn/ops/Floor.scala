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

class Floor[T: ClassTag]()
  (implicit ev: TensorNumeric[T]) extends Operation[Tensor[T], T] {

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output.resizeAs(input)
    input.getType() match {
      case FloatType =>
        output.asInstanceOf[Tensor[Float]].applyFun[Float](
          input.asInstanceOf[Tensor[Float]],
          scala.math.floor(_).asInstanceOf[Float])
      case DoubleType =>
        output.asInstanceOf[Tensor[Double]].applyFun[Double](
          input.asInstanceOf[Tensor[Double]],
          scala.math.floor)
      case _ => throw new RuntimeException("Unsupported tensor type")
    }

    output
  }
}

object Floor {
  def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]): Operation[Tensor[T], T]
  = ModuleToOperation[Tensor[T], T](new Floor())
}