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

import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.nn.tf.BiasAdd
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

class BiasAddGrad[T: ClassTag](dataFormat: DataFormat)
  (implicit ev: TensorNumeric[T])
  extends Operation[Tensor[T], Tensor[T], T] {

  private val module = BiasAdd()

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    getBiasDims(input)
    output.resizeAs(input).copy(input)
    dataFormat match {
      case DataFormat.NCHW =>
        output = output.resize(Array(batch, channel, height, width)).sum(1)
        output = output.sum(3)
        output = output.sum(4)
      case DataFormat.NHWC =>
        output = output.resize(Array(batch * height * width, channel)).sum(1)
    }
    output
  }

  private var batch : Int = 1
  private var channel : Int = 1
  private var width : Int = 1
  private var height : Int = 1

  private def getBiasDims(tensor: Tensor[_]): Unit = {
    batch = 1
    channel = 1
    width = 1
    height = 1
    dataFormat match {
      case DataFormat.NHWC =>
        val channelDim = tensor.dim()
        channel = tensor.size(channelDim)
        var i = 1
        while(i < channelDim) {
          batch *= tensor.size(i)
          i += 1
        }
      case DataFormat.NCHW =>
        val channelDim = tensor.dim() - 2
        val heightDim = tensor.dim() - 1
        val widthDim = tensor.dim()
        channel = tensor.size(channelDim)
        height = tensor.size(heightDim)
        width = tensor.size(widthDim)
        var i = 1
        while(i < channelDim) {
          batch *= tensor.size(i)
          i += 1
        }
    }
  }
}

object BiasAddGrad {
  def apply[T: ClassTag](dataFormat: DataFormat)
    (implicit ev: TensorNumeric[T]): BiasAddGrad[T] = new BiasAddGrad(dataFormat)
}
