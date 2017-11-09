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

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/**
 * Given shapes of two tensors, computes the reduction indices for the
 * gradient computation.
 *
 * @tparam T Numeric type. Only support float/double now
 */
class BroadcastGradientArgs[T: ClassTag]()
  (implicit ev: TensorNumeric[T]) extends Operation[Table, Table, T] {

  override def updateOutput(input: Table): Table = {
    val input1 = input[Tensor[Int]](1)
    val input2 = input[Tensor[Int]](2)

    val output1 = Tensor[Int]()
    val output2 = Tensor[Int]()

    output.insert(output1).insert(output2)

    // Reverse the shape of x and y for convenience.
    // After the reverse, 0-th is the inner-most dimension.
    val rx =
      if (input1.storage() == null) Array[Int]().toBuffer
      else input1.storage().array().reverse.toBuffer
    val ry =
      if (input2.storage() == null) Array[Int]().toBuffer
      else input2.storage().array().reverse.toBuffer

    if (rx.length < ry.length) {
      while (rx.length < ry.length) {
        rx.append(1)
      }
    } else {
      while (rx.length > ry.length) {
        ry.append(1)
      }
    }

    val xReducedIndexBuffer = new ArrayBuffer[Int]()
    val yReducedIndexBuffer = new ArrayBuffer[Int]()

    val n = rx.length

    var i = 0
    while (i < n) {
      val xi = rx(i)
      val yi = ry(i)

      if (xi == yi) {
        if (xi == 1) {
          xReducedIndexBuffer.append(n - 1 - i)
          yReducedIndexBuffer.append(n - 1 - i)
        }
      } else if (xi == 1) {
        xReducedIndexBuffer.append(n - 1 - i)
      } else if (yi == 1) {
        yReducedIndexBuffer.append(n - 1 - i)
      } else {
        return output
      }
      i += 1
    }

    if (xReducedIndexBuffer.isEmpty) {
      input(1) = Tensor[Int]()
    } else {
      output1.resize(Array(xReducedIndexBuffer.length))
        .set(Tensor[Int](Storage(xReducedIndexBuffer.reverse.toArray)))
    }

    if (yReducedIndexBuffer.isEmpty) {
      input(2) = Tensor[Int]()
    } else {
      output2.resize(Array(yReducedIndexBuffer.length))
        .set(Tensor[Int](Storage(yReducedIndexBuffer.reverse.toArray)))
    }

    output
  }
}

object BroadcastGradientArgs {
  def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]): Operation[Activity, Activity, T]
  = ModuleToOperation[T](new BroadcastGradientArgs())
}
