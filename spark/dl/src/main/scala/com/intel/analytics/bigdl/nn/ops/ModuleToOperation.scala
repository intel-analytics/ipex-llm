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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * Wrap a nn models to an [[Operation]], if an nn module's function
 * exactly corresponds to an Tensoflow operation.
 *
 * @param module an nn module
 * @tparam A Input data type
 * @tparam T Numeric type. Only support float/double now
 */
class ModuleToOperation[A <: Activity: ClassTag, T: ClassTag]
(module: AbstractModule[A, Tensor[T], T])
  (implicit ev: TensorNumeric[T])
  extends Operation[A, T]{

  override def updateOutput(input: A): Tensor[T] = {
    output = module.forward(input)
    output
  }
}

object ModuleToOperation {
  def apply[A <: Activity: ClassTag, T: ClassTag](model: AbstractModule[A, Tensor[T], T])
    (implicit ev: TensorNumeric[T]): ModuleToOperation[A, T] = new ModuleToOperation(model)
}
