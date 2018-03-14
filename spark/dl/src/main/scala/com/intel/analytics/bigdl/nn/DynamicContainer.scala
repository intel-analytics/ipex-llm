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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.ops.Operation
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Util

import scala.reflect.ClassTag

/**
 * DynamicContainer allow user to change its submodules after create it.
 * @tparam A Input data type
 * @tparam B Output data type
 * @tparam T Numeric type. Only support float/double now
 */
abstract class DynamicContainer[A <: Activity : ClassTag, B <: Activity : ClassTag, T: ClassTag](
  implicit ev: TensorNumeric[T]) extends Container[A, B, T] {

  /**
   * Add a sub-module to the contained `modules`
   *
   * @param module module to be add
   * @return this container
   */
  def add(module: AbstractModule[_ <: Activity, _ <: Activity, T]): this.type = {
    require(!module.isInstanceOf[Operation[_, _, _]],
      "Add operations to dynamic container is not allowed, as operations don't have backward. " +
        "Operation can only be used in Graph")
    validateInput[T](Seq(module))
    modules += module.asInstanceOf[AbstractModule[Activity, Activity, T]]
    checkDuplicate()
    this
  }
}
