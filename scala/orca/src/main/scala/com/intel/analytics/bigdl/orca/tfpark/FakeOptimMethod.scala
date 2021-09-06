/*
 * Copyright 2018 Analytics Zoo Authors.
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
package com.intel.analytics.zoo.tfpark

import com.intel.analytics.bigdl.optim.OptimMethod
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

class FakeOptimMethod[@specialized(Float, Double) T: ClassTag]()(implicit ev: TensorNumeric[T])
  extends OptimMethod[T] {

  override def optimize(feval: (Tensor[T]) =>
    (T, Tensor[T]), parameter: Tensor[T]): (Tensor[T], Array[T]) = {
    val (fx, dfdx) = feval(parameter)
    parameter.copy(dfdx)
    (parameter, Array(fx))
  }

  override def clearHistory(): Unit = {
  }

  override def getLearningRate(): Double = {
    1.0
  }

  override def loadFromTable(config: Table): this.type = {
    this
  }
}
