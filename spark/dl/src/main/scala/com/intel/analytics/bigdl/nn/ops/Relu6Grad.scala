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
import com.intel.analytics.bigdl.nn.{ReLU6 => ReLU6Layer}

import scala.reflect.ClassTag

class Relu6Grad[T: ClassTag](implicit ev: TensorNumeric[T])
  extends UnaryGrad[T](true) {

  val module: Module = ReLU6Layer[T]()
}

object Relu6Grad {
  def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]): Relu6Grad[T] = new Relu6Grad()
}
