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

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

class Assign[T: ClassTag](
  validateShape: Boolean = true,
  useLocking: Boolean = true
)
  (implicit ev: TensorNumeric[T]) extends Operation[Table, Tensor[_], T] {

  override def updateOutput(input: Table): Tensor[_] = {
    val input1 = input[Tensor[_]](1)
    val input2 = input[Tensor[_]](2)

    require(input1.getType() == input2.getType(),
      "ref and value must have the same tensor numeric type")

    if (validateShape) {
      var i = 1
      while (i <= input1.dim()) {
        require(input1.size(i) == input2.size(i), "shape of the ref and value are not same")
        i += 1
      }
    }


    input1.getType() match {
      case FloatType =>
        if (output.getType() != FloatType) {
          output = Activity.allocate[Tensor[Float], Float]()
        }
        val ref = input1.asInstanceOf[Tensor[Float]]
        val value = input2.asInstanceOf[Tensor[Float]]
        ref
          .resizeAs(value)
          .copy(value)
        output.asInstanceOf[Tensor[Float]]
          .resizeAs(value)
          .copy(value)
      case BooleanType =>
        if (output.getType() != FloatType) {
          output = Activity.allocate[Tensor[Float], Float]()
        }
        val ref = input1.asInstanceOf[Tensor[Float]]
        val value = input2.asInstanceOf[Tensor[Float]]
        ref
          .resizeAs(value)
          .copy(value)
        output.asInstanceOf[Tensor[Float]]
          .resizeAs(value)
          .copy(value)
      case DoubleType =>
        if (output.getType() != FloatType) {
          output = Activity.allocate[Tensor[Double], Double]()
        }
        val ref = input1.asInstanceOf[Tensor[Double]]
        val value = input2.asInstanceOf[Tensor[Double]]
        ref
          .resizeAs(value)
          .copy(value)
        output.asInstanceOf[Tensor[Double]]
          .resizeAs(value)
          .copy(value)
      case CharType =>
        if (output.getType() != FloatType) {
          output = Activity.allocate[Tensor[Char], Char]()
        }
        val ref = input1.asInstanceOf[Tensor[Char]]
        val value = input2.asInstanceOf[Tensor[Char]]
        ref
          .resizeAs(value)
          .copy(value)
        output.asInstanceOf[Tensor[Char]]
          .resizeAs(value)
          .copy(value)
      case StringType =>
        if (output.getType() != FloatType) {
          output = Activity.allocate[Tensor[String], String]()
        }
        val ref = input1.asInstanceOf[Tensor[String]]
        val value = input2.asInstanceOf[Tensor[String]]
        ref
          .resizeAs(value)
          .copy(value)
        output.asInstanceOf[Tensor[String]]
          .resizeAs(value)
          .copy(value)
      case LongType =>
        if (output.getType() != FloatType) {
          output = Activity.allocate[Tensor[Long], Long]()
        }
        val ref = input1.asInstanceOf[Tensor[Long]]
        val value = input2.asInstanceOf[Tensor[Long]]
        ref
          .resizeAs(value)
          .copy(value)
        output.asInstanceOf[Tensor[Long]]
          .resizeAs(value)
          .copy(value)
      case ShortType =>
        if (output.getType() != FloatType) {
          output = Activity.allocate[Tensor[Short], Short]()
        }
        val ref = input1.asInstanceOf[Tensor[Short]]
        val value = input2.asInstanceOf[Tensor[Short]]
        ref
          .resizeAs(value)
          .copy(value)
        output.asInstanceOf[Tensor[Short]]
          .resizeAs(value)
          .copy(value)
      case IntType =>
        if (output.getType() != FloatType) {
          output = Activity.allocate[Tensor[Int], Int]()
        }
        val ref = input1.asInstanceOf[Tensor[Int]]
        val value = input2.asInstanceOf[Tensor[Int]]
        ref
          .resizeAs(value)
          .copy(value)
        output.asInstanceOf[Tensor[Int]]
          .resizeAs(value)
          .copy(value)
      case _ => throw new RuntimeException("Unsupported tensor type")
    }

    output
  }
}

object Assign {
  def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]): Operation[Activity, Activity, T]
  = ModuleToOperation[T](new Assign())
}
