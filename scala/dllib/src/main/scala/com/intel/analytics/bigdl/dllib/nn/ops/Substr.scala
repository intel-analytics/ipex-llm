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

import com.google.protobuf.ByteString
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

class Substr[T: ClassTag]()
   (implicit ev: TensorNumeric[T]) extends Operation[Table, Tensor[ByteString], T] {

  override def updateOutput(input: Table): Tensor[ByteString] = {
    val data = input[Tensor[ByteString]](1).value()
    val pos = input[Tensor[Int]](2).value()
    val len = input[Tensor[Int]](3).value()
    import com.intel.analytics.bigdl.utils.tf.TFTensorNumeric.NumericByteString

    output = Tensor.scalar(data.substring(pos, pos + len))
    output
  }
}

object Substr {
  def apply[T: ClassTag]()
                        (implicit ev: TensorNumeric[T]):
  Operation[Activity, Activity, T]
  = new Substr[T]().asInstanceOf[Operation[Activity, Activity, T]]
}
