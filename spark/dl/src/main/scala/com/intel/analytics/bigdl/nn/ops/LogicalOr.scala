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
import com.intel.analytics.bigdl.tensor.{BooleanType, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

class LogicalOr[T: ClassTag]()
  (implicit ev: TensorNumeric[T]) extends Operation[Table, Tensor[Boolean], T] {

  output = Activity.allocate[Tensor[Boolean], Boolean]()

  override def updateOutput(input: Table): Tensor[Boolean] = {
    output.resizeAs(input(1)).copy(input(1))
    input[Tensor[_]](1).getType() match {
      case BooleanType =>
        output.map(input(2).asInstanceOf[Tensor[Boolean]], (a, b) => a || b)
      case _ => throw new RuntimeException("LogicalOr only support boolean tensor")
    }

    output
  }
}

object LogicalOr {
  def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]): Operation[Activity, Activity, T]
  = ModuleToOperation[T](new LogicalOr())
}
