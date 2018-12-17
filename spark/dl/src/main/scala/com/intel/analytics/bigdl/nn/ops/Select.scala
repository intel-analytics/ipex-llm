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

import scala.reflect.ClassTag

/**
 * Selects elements from input, depending on given condition. The input is a table (condition, t, e)
 * @tparam T Numeric type. Only support float/double now
 */
class Select[T: ClassTag]()
  (implicit ev: TensorNumeric[T]) extends Operation[Table, Activity, T] {

  override def updateOutput(input: Table): Activity = {
    val condition = input[Tensor[Boolean]](1)
    require(condition.isScalar, "only support condition as a scalar")
    val t = input[Activity](2)
    val e = input[Activity](3)

    output = if (condition.value()) t else e
    output
  }
}

object Select {
  def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]): Select[T] = new Select[T]()
}
