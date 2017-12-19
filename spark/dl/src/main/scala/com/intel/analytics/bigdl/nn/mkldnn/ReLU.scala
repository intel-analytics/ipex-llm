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

package com.intel.analytics.bigdl.nn.mkldnn

import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.mkl
import com.intel.analytics.bigdl.mkl.MklDnn
import com.intel.analytics.bigdl.mkl.MklDnn.EngineType

import scala.reflect.ClassTag

class ReLU[T: ClassTag](ip: Boolean = false)(
  implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  MklDnn.isLoaded

  MklDnn.EngineCreate(EngineType.cpu, 0)

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output = input
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput = gradOutput
    gradInput
  }
}

object ReLU {
  def apply[T: ClassTag](ip: Boolean = false)(implicit ev: TensorNumeric[T]): ReLU[T] = {
    new ReLU[T](ip)
  }
}
