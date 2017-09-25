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
import com.intel.analytics.bigdl.tensor.TensorNumericMath.{NumericWildCard, TensorNumeric}
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

class BiasAdd[T: ClassTag]()
  (implicit ev: TensorNumeric[T]) extends Operation[Table, Tensor[_], T] {
  var onesBias: Tensor[NumericWildCard] = _

  override def updateOutput(input: Table): Tensor[_] = {
    val value = input[Tensor[NumericWildCard]](1)
    val bias = input[Tensor[NumericWildCard]](2)
    val sizes = value.size().toBuffer
    val last = sizes.last
    sizes.remove(value.nDimension() - 1)
    val sizeProduct = sizes.product

    if (value.getType() != output.getType()) {
      output = value.emptyInstance()
    }

    if (onesBias == null) {
      onesBias = value.emptyInstance()
    }

    if (onesBias.dim() != 1 || onesBias.size(1) != sizeProduct) {
      onesBias.resize(sizeProduct).fill(ev.fromType(1.0))
    }

    output.asInstanceOf[Tensor[NumericWildCard]]
      .resizeAs(value)
      .copy(value)
    val value2d = output
      .view(Array(sizeProduct, last))
      .asInstanceOf[Tensor[NumericWildCard]]


    value2d
      .addr(
        value.getTensorNumeric().one,
        onesBias,
        bias)

    output
  }
}

object BiasAdd {
  def apply[T: ClassTag]()
    (implicit ev: TensorNumeric[T]):
  Operation[Activity, Activity, T]
  = ModuleToOperation[T](new BiasAdd())
}
