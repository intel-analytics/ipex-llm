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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * [[Operation]] is an abstract class which represents a forward only layer.
 * An operations has only forward functions and without backward functions.
 * An operations should be only used in graph and make sure the backward graph won't contain
 * operations.
 *
 * @tparam A Input data type
 * @tparam T Numeric type. Only support float/double now
 */
abstract class Operation[A <: Activity: ClassTag, B <: Activity: ClassTag, T: ClassTag]
(implicit ev: TensorNumeric[T]) extends AbstractModule[A, B, T]{

  gradInput = Activity.emptyGradInput(this.getName()).asInstanceOf[A]

  final override def updateGradInput(input: A, gradOutput: B): A = {
    throw new UnsupportedOperationException("Operation does not support updateGradInput() method")
  }

  final override def backward(input: A, gradOutput: B): A = {
    throw new UnsupportedOperationException("Operation does not support backward() method")
  }
}
