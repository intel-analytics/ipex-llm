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
package com.intel.analytics.bigdl.example.tensorflow.loadandsave

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.utils.tf.TensorflowSaver

/**
 * This example shows how to define a BigDL model and save it as tensorflow format
 */
object Save {
  def main(args: Array[String]) {
    val model = lenet()
    TensorflowSaver.saveGraph(model, Seq(("input", Seq(1, 1, 28, 28))), "./bigdl.pb")
  }

  def lenet(): Graph[Float] = {
    val conv1 = SpatialConvolution(1, 6, 5, 5).setName("conv1_5x5").inputs()
    val tanh1 = Tanh().setName("tanh1").inputs(conv1)
    val pool1 = SpatialMaxPooling(2, 2, 2, 2).setName("pool1").inputs(tanh1)
    val tanh2 = Tanh().setName("tanh2").inputs(pool1)
    val conv2 = SpatialConvolution(6, 12, 5, 5).setName("conv2_5x5").inputs(tanh2)
    val pool2 = SpatialMaxPooling(2, 2, 2, 2).setName("pool2").inputs(conv2)
    val reshape2 = Reshape(Array(1, 12 * 4 * 4)).setName("reshape2").inputs(pool2)
    val fc1 = Linear(12 * 4 * 4, 100).setName("fc1").inputs(reshape2)
    val tanh3 = Tanh().setName("tanh3").inputs(fc1)
    val fc2 = Linear(100, 10).setName("fc2").inputs(tanh3)
    val output = LogSoftMax().setName("output").inputs(fc2)
    Graph(conv1, output)
  }
}
