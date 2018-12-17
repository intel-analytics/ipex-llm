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
import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.mkldnn.DnnGraph

object LeNet5 {
  def apply(classNum: Int): Module[Float] = {
    val model = Sequential()
    model.add(Reshape(Array(1, 28, 28)))
      .add(SpatialConvolution(1, 6, 5, 5).setName("conv1_5x5"))
      .add(Tanh())
      .add(SpatialMaxPooling(2, 2, 2, 2))
      .add(SpatialConvolution(6, 12, 5, 5).setName("conv2_5x5"))
      .add(Tanh())
      .add(SpatialMaxPooling(2, 2, 2, 2))
      .add(Reshape(Array(12 * 4 * 4)))
      .add(Linear(12 * 4 * 4, 100).setName("fc1"))
      .add(Tanh())
      .add(Linear(100, classNum).setName("fc2"))
      .add(LogSoftMax())
  }

  def graph(classNum: Int): Module[Float] = {
    val input = Reshape(Array(1, 28, 28)).inputs()
    val conv1 = SpatialConvolution(1, 6, 5, 5).setName("conv1_5x5").inputs(input)
    val tanh1 = Tanh().inputs(conv1)
    val pool1 = SpatialMaxPooling(2, 2, 2, 2).inputs(tanh1)
    val conv2 = SpatialConvolution(6, 12, 5, 5).setName("conv2_5x5").inputs(pool1)
    val tanh2 = Tanh().inputs(conv2)
    val pool2 = SpatialMaxPooling(2, 2, 2, 2).inputs(tanh2)
    val reshape = Reshape(Array(12 * 4 * 4)).inputs(pool2)
    val fc1 = Linear(12 * 4 * 4, 100).setName("fc1").inputs(reshape)
    val tanh3 = Tanh().inputs(fc1)
    val fc2 = Linear(100, classNum).setName("fc2").inputs(tanh3)
    val output = LogSoftMax().inputs(fc2)

    Graph(input, output)
  }

  def keras(classNum: Int): nn.keras.Sequential[Float] = {
    import com.intel.analytics.bigdl.nn.keras._
    import com.intel.analytics.bigdl.utils.Shape

    val model = Sequential()
    model.add(Reshape(Array(1, 28, 28), inputShape = Shape(28, 28, 1)))
    model.add(Convolution2D(6, 5, 5, activation = "tanh").setName("conv1_5x5"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(12, 5, 5, activation = "tanh").setName("conv2_5x5"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(100, activation = "tanh").setName("fc1"))
    model.add(Dense(classNum, activation = "softmax").setName("fc2"))
  }

  def kerasGraph(classNum: Int): nn.keras.Model[Float] = {
    import com.intel.analytics.bigdl.nn.keras._
    import com.intel.analytics.bigdl.utils.Shape

    val input = Input(inputShape = Shape(28, 28, 1))
    val reshape = Reshape(Array(1, 28, 28)).inputs(input)
    val conv1 = Convolution2D(6, 5, 5, activation = "tanh").setName("conv1_5x5").inputs(reshape)
    val pool1 = MaxPooling2D().inputs(conv1)
    val conv2 = Convolution2D(12, 5, 5, activation = "tanh").setName("conv2_5x5").inputs(pool1)
    val pool2 = MaxPooling2D().inputs(conv2)
    val flatten = Flatten().inputs(pool2)
    val fc1 = Dense(100, activation = "tanh").setName("fc1").inputs(flatten)
    val fc2 = Dense(classNum, activation = "softmax").setName("fc2").inputs(fc1)
    Model(input, fc2)
  }

  def dnn(batchSize: Int, classNum: Int): mkldnn.Sequential = {
    val inputShape = Array(batchSize, 1, 28, 28)
    val outputShape = Array(batchSize, 10)

    val model = mkldnn.Sequential()
      .add(mkldnn.Input(inputShape, Memory.Format.nchw))
      .add(mkldnn.SpatialConvolution(1, 20, 5, 5).setName("conv1"))
      .add(mkldnn.SpatialBatchNormalization(20).setName("bn1"))
      .add(mkldnn.MaxPooling(2, 2, 2, 2).setName("pool1"))
      .add(mkldnn.SpatialConvolution(20, 50, 5, 5).setName("conv2"))
      .add(mkldnn.MaxPooling(2, 2, 2, 2).setName("pool2"))
      .add(mkldnn.Linear(50 * 4 * 4, 500).setName("ip1"))
      .add(mkldnn.ReLU().setName("relu1"))
      .add(mkldnn.Linear(500, 10).setName("ip2"))
      .add(mkldnn.ReorderMemory(mkldnn.HeapData(outputShape, Memory.Format.nc)))
    model
  }

  def dnnGraph(batchSize: Int, classNum: Int): mkldnn.DnnGraph = {
    val inputShape = Array(batchSize, 1, 28, 28)
    val outputShape = Array(batchSize, 10)

    val input = mkldnn.Input(inputShape, Memory.Format.nchw).inputs()
    val conv1 = mkldnn.SpatialConvolution(1, 20, 5, 5).setName("conv1").inputs(input)
    val bn1 = mkldnn.SpatialBatchNormalization(20).setName("bn1").inputs(conv1)
    val pool1 = mkldnn.MaxPooling(2, 2, 2, 2).setName("pool1").inputs(bn1)
    val conv2 = mkldnn.SpatialConvolution(20, 50, 5, 5).setName("conv2").inputs(pool1)
    val pool2 = mkldnn.MaxPooling(2, 2, 2, 2).setName("pool2").inputs(conv2)
    val ip1 = mkldnn.Linear(50 * 4 * 4, 500).setName("ip1").inputs(pool2)
    val relu1 = mkldnn.ReLU().setName("relu1").inputs(ip1)
    val ip2 = mkldnn.Linear(500, 10).setName("ip2").inputs(relu1)
    val output = mkldnn.ReorderMemory(mkldnn.HeapData(outputShape, Memory.Format.nc)).inputs(ip2)

    DnnGraph(Array(input), Array(output))
  }
}
