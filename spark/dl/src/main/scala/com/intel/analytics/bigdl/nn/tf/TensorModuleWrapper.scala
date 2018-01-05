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
package com.intel.analytics.bigdl.nn.tf

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, TensorModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * The AbstractModule[Tensor[D], Tensor[D], T] that takes a TensorModule[D] to
 * enable the computation of D typed data in a module with numeric type of T.
 */
class TensorModuleWrapper[T: ClassTag, D: ClassTag] private
(val module: TensorModule[D])
(implicit ev: TensorNumeric[T], evd: TensorNumeric[D])
extends AbstractModule[Tensor[D], Tensor[D], T] {

  output = Tensor[D]()
  gradInput = Tensor[D]()

  override def updateOutput(input: Tensor[D]): Tensor[D] = {
    output = module.forward(input)
    output
  }

  override def updateGradInput(input: Tensor[D], gradOutput: Tensor[D]): Tensor[D] = {
    module.backward(input, gradOutput)
  }

  override def getClassTagNumerics() : (Array[ClassTag[_]], Array[TensorNumeric[_]]) = {
    (Array[ClassTag[_]](scala.reflect.classTag[T], scala.reflect.classTag[D]),
      Array[TensorNumeric[_]](ev, evd))
  }
}

object TensorModuleWrapper {
  def apply[T: ClassTag, D: ClassTag](model: TensorModule[D])
     (implicit ev: TensorNumeric[T], evd: TensorNumeric[D]): TensorModuleWrapper[T, D] =
    new TensorModuleWrapper[T, D](model)
}
