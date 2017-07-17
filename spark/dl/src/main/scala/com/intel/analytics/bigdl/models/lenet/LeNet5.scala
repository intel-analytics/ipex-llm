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

package com.intel.analytics.bigdl.models.lenet

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.DataFormat

object LeNet5 {
  def apply(classNum: Int, format: DataFormat = DataFormat.NCHW): Module[Float] = {
    val inputLayer = format match {
      case DataFormat.NCHW => Reshape(Array(1, 28, 28))
      case DataFormat.NHWC => Reshape(Array(28, 28, 1))
    }
    val model = Sequential()
    model.add(inputLayer)
      .add(SpatialConvolution(1, 6, 5, 5, format = format).setName("conv1_5x5"))
      .add(Tanh())
      .add(SpatialMaxPooling(2, 2, 2, 2, format = format))
      .add(Tanh())
      .add(SpatialConvolution(6, 12, 5, 5, format = format).setName("conv2_5x5"))
      .add(SpatialMaxPooling(2, 2, 2, 2, format = format))
      .add(Reshape(Array(12 * 4 * 4)))
      .add(Linear(12 * 4 * 4, 100).setName("fc1"))
      .add(Tanh())
      .add(Linear(100, classNum).setName("fc2"))
      .add(LogSoftMax())
  }
}
