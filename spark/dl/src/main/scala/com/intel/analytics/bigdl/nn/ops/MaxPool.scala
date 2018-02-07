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

import com.intel.analytics.bigdl.nn.SpatialMaxPooling
import com.intel.analytics.bigdl.nn.abstractnn.{Activity, DataFormat}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor._

import scala.reflect.ClassTag

class MaxPool[T: ClassTag](
  val ksize: Array[Int],
  val strides: Array[Int],
  val padding: String,
  val format: DataFormat = DataFormat.NHWC
)(implicit ev: TensorNumeric[T]) extends Operation[Tensor[T], Tensor[T], T] {
  val pool: SpatialMaxPooling[T] = format match {
    case DataFormat.NHWC =>
      if (padding == "SAME") {
        SpatialMaxPooling(
          kH = ksize(1),
          kW = ksize(2),
          dH = strides(1),
          dW = strides(2),
          padH = -1,
          padW = -1,
          format = format
        )
      } else if (padding == "VALID") {
        SpatialMaxPooling(
          kH = ksize(1),
          kW = ksize(2),
          dH = strides(1),
          dW = strides(2),
          format = format
        )
      } else {
        throw new RuntimeException("Padding can only support SAME and VALID padding")
      }
    case DataFormat.NCHW =>
      if (padding == "SAME") {
        SpatialMaxPooling(
          kH = ksize(2),
          kW = ksize(3),
          dH = strides(2),
          dW = strides(3),
          padH = -1,
          padW = -1,
          format = format
        )
      } else if (padding == "VALID") {
        SpatialMaxPooling(
          kH = ksize(2),
          kW = ksize(3),
          dH = strides(2),
          dW = strides(3),
          format = format
        )
      } else {
        throw new RuntimeException("Padding can only support SAME and VALID padding")
      }
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output = pool.updateOutput(input)
    output
  }
}

object MaxPool {
  def apply[T: ClassTag](
    ksize: Array[Int],
    strides: Array[Int],
    padding: String,
    format: DataFormat = DataFormat.NHWC
  )(implicit ev: TensorNumeric[T]): Operation[Activity, Activity, T]
  = ModuleToOperation[T](new MaxPool(ksize, strides, padding, format))
}
