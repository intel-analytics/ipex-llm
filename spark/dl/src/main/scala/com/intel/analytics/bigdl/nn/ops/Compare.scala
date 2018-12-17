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

abstract class Compare[T: ClassTag]()
(implicit ev: TensorNumeric[T]) extends Operation[Table, Tensor[Boolean], T] {

  def compareFloat(a: Float, b: Float): Boolean

  def compareDouble(a: Double, b: Double): Boolean

  def compareChar(a: Char, b: Char): Boolean

  def compareLong(a: Long, b: Long): Boolean

  def compareShort(a: Short, b: Short): Boolean

  def compareInt(a: Int, b: Int): Boolean

  def compareBoolean(a: Boolean, b: Boolean): Boolean

  def compareByteString(a: ByteString, b: ByteString): Boolean

  output = Activity.allocate[Tensor[Boolean], Boolean]()

  override def updateOutput(input: Table): Tensor[Boolean] = {
    output.resizeAs(input(1))
    input[Tensor[_]](1).getType() match {
      case FloatType =>
        output.zipWith[Float, Float](
          input[Tensor[Float]](1),
          input[Tensor[Float]](2),
          (a, b) => compareFloat(a, b))
      case DoubleType =>
        output.zipWith[Double, Double](
          input[Tensor[Double]](1),
          input[Tensor[Double]](2),
          (a, b) => compareDouble(a, b))
      case CharType =>
        output.zipWith[Char, Char](
          input[Tensor[Char]](1),
          input[Tensor[Char]](2),
          (a, b) => compareChar(a, b))
      case LongType =>
        output.zipWith[Long, Long](
          input[Tensor[Long]](1),
          input[Tensor[Long]](2),
          (a, b) => compareLong(a, b))
      case ShortType =>
        output.zipWith[Short, Short](
          input[Tensor[Short]](1),
          input[Tensor[Short]](2),
          (a, b) => compareShort(a, b))
      case IntType =>
        output.zipWith[Int, Int](
          input[Tensor[Int]](1),
          input[Tensor[Int]](2),
          (a, b) => compareInt(a, b))
      case BooleanType =>
        output.zipWith[Boolean, Boolean](
          input[Tensor[Boolean]](1),
          input[Tensor[Boolean]](2),
          (a, b) => compareBoolean(a, b))
      case StringType =>
        output.zipWith[ByteString, ByteString](
          input[Tensor[ByteString]](1),
          input[Tensor[ByteString]](2),
          (a, b) => compareByteString(a, b))
      case _ => throw new RuntimeException("Unsupported tensor type")
    }

    output
  }
}
