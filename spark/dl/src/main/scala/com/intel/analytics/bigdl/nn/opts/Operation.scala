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
package com.intel.analytics.bigdl.nn.opts

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

abstract class Operation[A <: Activity: ClassTag,
@specialized(Float, Double) T: ClassTag](implicit ev: TensorNumeric[T])
  extends AbstractModule[A, Tensor[T], T]{

 override def updateGradInput(input: A, gradOutput: Tensor[T]): A = {
  throw new UnsupportedOperationException("Operation does not support updateGradInput() method")
 }

 override def backward(input: A, gradOutput: Tensor[T]): A = {
  throw new UnsupportedOperationException("Operation does not support backward() method")
 }
}
