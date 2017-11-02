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
import com.intel.analytics.bigdl.nn.{ELU => ELULayer}

import scala.reflect.ClassTag

class EluGrad[T: ClassTag, D: ClassTag]
(implicit ev: TensorNumeric[T], ev2: TensorNumeric[D])
  extends UnaryGrad[T, D](true, true) {

  override val module: Module = ELULayer[T, D]()
}

object EluGrad {
  def apply[T: ClassTag, D: ClassTag]()
      (implicit ev: TensorNumeric[T], ev2: TensorNumeric[D]): EluGrad[T, D] = new EluGrad[T, D]()
}
