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
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

class Mod[T: ClassTag, D: ClassTag]()(implicit ev: TensorNumeric[T], ev2: TensorNumeric[D])
  extends Operation[Table, Tensor[D], T]{

  output = Tensor[D]()
  private val buffer = Tensor[D]()

  override def updateOutput(input: Table): Tensor[D] = {
    val input1 = input[Tensor[D]](1)
    val input2 = input[Tensor[D]](2)
    output.resizeAs(input1).copy(input1)
    buffer.resizeAs(output).copy(output)
    buffer.div(input2).apply1(ev2.truncate(_)).cmul(input2)
    output.sub(buffer)
  }

  override def clearState(): Mod.this.type = {
    super.clearState()
    buffer.set()
    this
  }

  override def getClassTagNumerics() : (Array[ClassTag[_]], Array[TensorNumeric[_]]) = {
    (Array[ClassTag[_]](scala.reflect.classTag[T], scala.reflect.classTag[D]),
      Array[TensorNumeric[_]](ev, ev2))
  }
}

object Mod {
  def apply[T: ClassTag, D: ClassTag]()(implicit ev: TensorNumeric[T], ev2: TensorNumeric[D])
  : Mod[T, D] = new Mod()
}
