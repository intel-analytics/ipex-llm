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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.{NumericWildCard, TensorNumeric}
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

class Pow[T: ClassTag]()
  (implicit ev: TensorNumeric[T]) extends Operation[Table, Tensor[_], T]{

  override def updateOutput(input: Table): Tensor[_] = {
    val v = input[Tensor[NumericWildCard]](2).value()
    val t = input[Tensor[NumericWildCard]](1)
    if (output.getType() != t.getType()) {
      output = t.emptyInstance()
    }
    output.resizeAs(t)
    output.asInstanceOf[Tensor[NumericWildCard]].pow(t, v)
  }
}

object Pow {
  def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]): Pow[T] = new Pow[T]()
}
