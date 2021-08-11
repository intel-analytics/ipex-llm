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

class RangeOps[T: ClassTag, D: ClassTag]()
  (implicit ev: TensorNumeric[T], ev2: TensorNumeric[D])
  extends Operation[Table, Tensor[D], T] {

  output = Tensor[D]()

  override def updateOutput(input: Table): Tensor[D] = {
    require(input.length() == 3, s"require 3 tensors as input, but get ${input.length()}")
    val start = input[Tensor[D]](1).value()
    val limit = input[Tensor[D]](2).value()
    val delta = input[Tensor[D]](3).value()

    output = Tensor[D](ev2.toType[Int](ev2.divide(ev2.minus(limit, start), delta)))
    var i = 0
    while(i < output.size(1)) {
      output.setValue(i + 1, ev2.plus(start, ev2.times(ev2.fromType(i), delta)))
      i += 1
    }
    output
  }

  override def getClassTagNumerics() : (Array[ClassTag[_]], Array[TensorNumeric[_]]) = {
    (Array[ClassTag[_]](scala.reflect.classTag[T], scala.reflect.classTag[D]),
      Array[TensorNumeric[_]](ev, ev2))
  }
}

object RangeOps {
  def apply[T: ClassTag, D: ClassTag]()(implicit ev: TensorNumeric[T], ev2: TensorNumeric[D])
  : RangeOps[T, D] = new RangeOps[T, D]()
}
