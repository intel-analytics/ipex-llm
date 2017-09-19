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

class BiasAdd[T: ClassTag]()
  (implicit ev: TensorNumeric[T]) extends Operation[Table, Tensor[_], T] {
  val onesBias = Tensor()

  override def updateOutput(input: Table): Tensor[_] = {
    val value = input[Tensor[_]](1)
    val bias = input[Tensor[_]](2)
    val sizes = value.size().toBuffer
    val last = sizes.last
    sizes.remove(value.nDimension() - 1)
    val sizeProduct = sizes.product

    if (onesBias.dim() != 1 || onesBias.size(1) != sizeProduct) {
      onesBias.resize(sizeProduct).fill(ev.fromType(1.0))
    }

    value.getType() match {
      case FloatType =>
        if (output.getType() != FloatType) {
          output = Activity.allocate[Tensor[Float], Float]()
        }
        output
          .asInstanceOf[Tensor[Float]]
          .resizeAs(value)
          .copy(value.asInstanceOf[Tensor[Float]])
        val value2d = output.view(Array(sizeProduct, last))

        value2d.asInstanceOf[Tensor[Float]]
          .addr(
            1f,
            onesBias.asInstanceOf[Tensor[Float]],
            bias.asInstanceOf[Tensor[Float]])
      case IntType =>
        if (output.getType() != IntType) {
          output = Activity.allocate[Tensor[Int], Int]()
        }
        output
          .asInstanceOf[Tensor[Int]]
          .resizeAs(value)
          .copy(value.asInstanceOf[Tensor[Int]])
        val value2d = output.view(Array(sizeProduct, last))

        value2d.asInstanceOf[Tensor[Int]]
          .addr(
            1,
            onesBias.asInstanceOf[Tensor[Int]],
            bias.asInstanceOf[Tensor[Int]])
      case DoubleType =>
        if (output.getType() != FloatType) {
          output = Activity.allocate[Tensor[Double], Double]()
        }
        output
          .asInstanceOf[Tensor[Double]]
          .resizeAs(value)
          .copy(value.asInstanceOf[Tensor[Double]])
        val value2d = output.view(Array(sizeProduct, last))

        value2d.asInstanceOf[Tensor[Double]]
          .addr(
            1.0,
            onesBias.asInstanceOf[Tensor[Double]],
            bias.asInstanceOf[Tensor[Double]])
      case LongType =>
        if (output.getType() != FloatType) {
          output = Activity.allocate[Tensor[Long], Long]()
        }
        output
          .asInstanceOf[Tensor[Long]]
          .resizeAs(value)
          .copy(value.asInstanceOf[Tensor[Long]])
        val value2d = output.view(Array(sizeProduct, last))

        value2d.asInstanceOf[Tensor[Long]]
          .addr(
            1L,
            onesBias.asInstanceOf[Tensor[Long]],
            bias.asInstanceOf[Tensor[Long]])
      case ShortType =>
        if (output.getType() != FloatType) {
          output = Activity.allocate[Tensor[Short], Short]()
        }
        output
          .asInstanceOf[Tensor[Short]]
          .resizeAs(value)
          .copy(value.asInstanceOf[Tensor[Short]])
        val value2d = output.view(Array(sizeProduct, last))

        value2d.asInstanceOf[Tensor[Short]]
          .addr(
            1.toShort,
            onesBias.asInstanceOf[Tensor[Short]],
            bias.asInstanceOf[Tensor[Short]])
      case _ => throw new RuntimeException("Unsupported tensor type")
    }

    output
  }
}

object BiasAdd {
  def apply[T: ClassTag]()
    (implicit ev: TensorNumeric[T]):
  Operation[Activity, Activity, T]
  = ModuleToOperation[T](new BiasAdd())
}
