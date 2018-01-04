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

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

abstract class UnaryGrad[T: ClassTag, D: ClassTag](
               gradFirst: Boolean = false,
               needForward: Boolean = false)
               (implicit ev: TensorNumeric[T], ev2: TensorNumeric[D])
  extends Operation[Table, Tensor[D], T]{

  type Module = AbstractModule[Tensor[D], Tensor[D], T]

  val module: Module

  override def updateOutput(input: Table): Tensor[D] = {
    val (grads, inputs) = if (gradFirst) {
      (input[Tensor[D]](1), input[Tensor[D]](2))
    } else {
      (input[Tensor[D]](2), input[Tensor[D]](1))
    }

    if (needForward) {
      module.forward(inputs)
    }

    output = module.updateGradInput(inputs, grads).toTensor[D]
    output
  }

  override def getClassTagNumerics() : (Array[ClassTag[_]], Array[TensorNumeric[_]]) = {
    (Array[ClassTag[_]](scala.reflect.classTag[T], scala.reflect.classTag[D]),
      Array[TensorNumeric[_]](ev, ev2))
  }
}
