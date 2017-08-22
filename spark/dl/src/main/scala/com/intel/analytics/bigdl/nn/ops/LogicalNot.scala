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
import com.intel.analytics.bigdl.tensor.{BooleanType, Tensor}

import scala.reflect.ClassTag

class LogicalNot[T: ClassTag]()
  (implicit ev: TensorNumeric[T]) extends Operation[Tensor[T], T] {
  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output.resizeAs(input).copy(input)
    ev.getType() match {
      case BooleanType =>
        output
          .toTensor[Boolean]
          .apply1(!_)
      case _ => throw new RuntimeException("LogicalAnd only support boolean tensor")
    }

    output
  }
}

object LogicalNot {
  def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]): Operation[Tensor[T], T]
  = ModuleToOperation[Tensor[T], T](new LogicalNot())
}