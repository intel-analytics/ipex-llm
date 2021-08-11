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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * Wrap a nn module to an [[Operation]]
 *
 * @param module an nn module
 * @tparam T Numeric type. Only support float/double now
 */
class ModuleToOperation[T: ClassTag]
(val module: AbstractModule[Activity, Activity, T])
  (implicit ev: TensorNumeric[T])
  extends Operation[Activity, Activity, T]{

  override def updateOutput(input: Activity): Activity = {
    output = module.forward(input)
    output
  }
}

object ModuleToOperation {
  def apply[T: ClassTag](model: AbstractModule[_, _, T])
    (implicit ev: TensorNumeric[T]): ModuleToOperation[T] =
    new ModuleToOperation(model.asInstanceOf[AbstractModule[Activity, Activity, T]])
}
