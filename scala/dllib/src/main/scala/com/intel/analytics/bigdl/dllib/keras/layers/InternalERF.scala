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

package com.intel.analytics.zoo.pipeline.api.keras.layers.internal

import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.common.MKLBlas

import scala.reflect.ClassTag

private[zoo] class InternalERF[T: ClassTag]()(
  implicit ev: TensorNumeric[T]) extends TensorModule[T] {
  val derivativeFactor = ev.fromType(1.1283791670955126)
  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output.resizeAs(input).copy(input)
    MKLBlas.erf(output)
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    val tensor = Tensor().resizeAs(input).copy(input)
    val derivative = (-tensor.pow(ev.fromType(2))).exp().mul(derivativeFactor)

    gradInput = gradOutput.cmul(derivative)
    gradInput
  }
}
