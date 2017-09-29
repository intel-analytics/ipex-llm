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
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

class NotEqual[T: ClassTag]()
  (implicit ev: TensorNumeric[T]) extends Operation[Table, Tensor[Boolean], T] {

  output = Activity.allocate[Tensor[Boolean], Boolean]()

  override def updateOutput(input: Table): Tensor[Boolean] = {
    output.resizeAs(input(1))
    input[Tensor[_]](1).getType() match {
      case FloatType =>
        output.zipWith[Float, Float](
          input[Tensor[Float]](1),
          input[Tensor[Float]](2),
          (a, b) => a != b)
      case BooleanType =>
        output.zipWith[Boolean, Boolean](
          input[Tensor[Boolean]](1),
          input[Tensor[Boolean]](2),
          (a, b) => a != b)
      case DoubleType =>
        output.zipWith[Double, Double](
          input[Tensor[Double]](1),
          input[Tensor[Double]](2),
          (a, b) => a != b)
      case CharType =>
        output.zipWith[Char, Char](
          input[Tensor[Char]](1),
          input[Tensor[Char]](2),
          (a, b) => a != b)
      case StringType =>
        output.zipWith[ByteString, ByteString](
          input[Tensor[ByteString]](1),
          input[Tensor[ByteString]](2),
          (a, b) => a != b)
      case LongType =>
        output.zipWith[Long, Long](
          input[Tensor[Long]](1),
          input[Tensor[Long]](2),
          (a, b) => a != b)
      case ShortType =>
        output.zipWith[Short, Short](
          input[Tensor[Short]](1),
          input[Tensor[Short]](2),
          (a, b) => a != b)
      case IntType =>
        output.zipWith[Int, Int](
          input[Tensor[Int]](1),
          input[Tensor[Int]](2),
          (a, b) => a != b)
      case _ => throw new RuntimeException("Unsupported tensor type")
    }

    output
  }
}

object NotEqual {
  def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]): Operation[Activity, Activity, T]
  = ModuleToOperation[T](new NotEqual())
}
