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

import com.google.protobuf.ByteString
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn.ops.Operation
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

/**
 * Assert will assert the first input to be true, if not, throw the message in the second
 * input. Assert has no output.
 */
private[bigdl] class Assert[T: ClassTag]()
  (implicit ev: TensorNumeric[T]) extends Operation[Table, Activity, T] {
  override def updateOutput(input: Table): Tensor[T] = {
    val predicateTensor = input(1).asInstanceOf[Tensor[Boolean]]
    val messageTensor = input(2).asInstanceOf[Tensor[ByteString]]

    val predicate = predicateTensor.value()
    val message = messageTensor.value()

    assert(predicate, message.toStringUtf8)
    null
  }
}
