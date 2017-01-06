/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.nn.dnn

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.nn.dnn.Tools._
import com.intel.analytics.bigdl.utils.{T, Table}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.collection.mutable.ArrayBuffer

class GoogLeNetV1Spec extends FlatSpec with BeforeAndAfter with Matchers {
  before {
    if (!Collect.hasCollect) {
      cancel("collect is not installed")
    }

    Affinity.acquireCore()
  }

  "An GoogLeNet_V1 " should "the same output, gradient as intelcaffe w/ dnn" in {
    val model = GoogleNet_v1(1000)
    model.convertToMklDnn()
    model.reset()

    val modules = ArrayBuffer[TensorModule[Float]]()
    Tools.flattenModules(model, modules)

    def iteration(): Unit = {
      val identity = Collect.run(googlenet_v1)
      val input = loadTensor("Fwrd_data_input", Array(batchSize, 3, 224, 224), identity)

      loadParameters(modules, identity)

      val output = model.forward(input).toTensor
      compareAllLayers(modules, identity, Forward)

      // start get outputs of each branch.
      val split1 = model.asInstanceOf[Sequential[Float]].modules(1)
      val output1 = split1.asInstanceOf[Concat[Float]].modules(1).asInstanceOf[Sequential[Float]]
      val mainBranch = split1.asInstanceOf[Concat[Float]].modules(0)
      val split2 = mainBranch.asInstanceOf[Sequential[Float]].modules(1)
      val output3 = split2.asInstanceOf[Concat[Float]].modules(0).asInstanceOf[Sequential[Float]]
      val output2 = split2.asInstanceOf[Concat[Float]].modules(1).asInstanceOf[Sequential[Float]]

      val last1 = output1.modules(output1.modules.length - 1)
      val last2 = output2.modules(output2.modules.length - 1)
      val last3 = output3.modules(output3.modules.length - 1)

      val loss1Output = last1.output.toTensor
      val loss2Output = last2.output.toTensor
      val loss3Output = last3.output.toTensor
      // end get outputs of each branch.

      val gradOutput3 = loadTensor("Bwrd_loss3_loss", loss3Output.size(), identity)
      val gradOutput2 = loadTensor("Bwrd_loss2_loss", loss2Output.size(), identity)
      val gradOutput1 = loadTensor("Bwrd_loss1_loss", loss1Output.size(), identity)

      // combine three gradOutputs
      val gradOutput = Tensor[Float](output.size())
      gradOutput.narrow(2, 1, gradOutput3.size(2)).copy(gradOutput3)
      gradOutput.narrow(2, gradOutput3.size(2) + 1, gradOutput2.size(2)).copy(gradOutput2)
      gradOutput.narrow(2, gradOutput2.size(2) * 2 + 1, gradOutput1.size(2)).copy(gradOutput1)

      val gradInput = model.backward(input, gradOutput).asInstanceOf[Tensor[Float]]

      compareAllLayers(modules, identity, Backward)

      // Check the gradInput, gradWeight, gradBias of first layer
      val firstLayerName = "Bwrd_" + modules(0).getName().replaceAll("/", "_")
      val gradInputCaffe = loadTensor(firstLayerName, gradInput.size(), identity)
      cumulativeError(gradInput, gradInputCaffe, "gradInput") should be(0.0)
      compareParameters(modules, identity)
    }

    for (i <- 0 until 5) {
      iteration()
    }
  }

  val batchSize = 4
  val googlenet_v1 = s"""
name: "GoogleNet"
force_backward: true
layer {
  name: "data_input"
  type: "DummyData"
  top: "data"
  include {
    phase: TRAIN
  }
  dummy_data_param {
    shape: { dim: $batchSize dim: 3 dim: 224 dim: 224 }
    data_filler {
#      type: "constant"
#      value: 0.01
      type: "uniform"
    }
  }
}
layer {
  name: "data_label"
  type: "DummyData"
  top: "label"
  include {
    phase: TRAIN
  }
  dummy_data_param {
    shape: { dim: $batchSize }
    data_filler {
      type: "constant"
    }
  }
}


layer {
  name: "conv1/7x7_s2"
  type: "Convolution"
  bottom: "data"
  top: "conv1/7x7_s2"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 64
    pad: 3
    kernel_size: 7
    stride: 2
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "conv1/relu_7x7"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "conv1/7x7_s2"
  top: "conv1/7x7_s2"
}
layer {
  name: "pool1/3x3_s2"
  type: "Pooling"
  bottom: "conv1/7x7_s2"
  top: "pool1/3x3_s2"
  pooling_param {
    engine: MKL2017
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "pool1/norm1"
  type: "LRN"
  bottom: "pool1/3x3_s2"
  top: "pool1/norm1"
  lrn_param {
    engine: MKL2017
    local_size: 5
    alpha: 0.0001
    beta: 0.75
  }
}
layer {
  name: "conv2/3x3_reduce"
  type: "Convolution"
  bottom: "pool1/norm1"
  top: "conv2/3x3_reduce"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 64
    kernel_size: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "conv2/relu_3x3_reduce"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "conv2/3x3_reduce"
  top: "conv2/3x3_reduce"
}
layer {
  name: "conv2/3x3"
  type: "Convolution"
  bottom: "conv2/3x3_reduce"
  top: "conv2/3x3"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 192
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "conv2/relu_3x3"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "conv2/3x3"
  top: "conv2/3x3"
}
layer {
  name: "conv2/norm2"
  type: "LRN"
  bottom: "conv2/3x3"
  top: "conv2/norm2"
  lrn_param {
    engine: MKL2017
    local_size: 5
    alpha: 0.0001
    beta: 0.75
  }
}
layer {
  name: "pool2/3x3_s2"
  type: "Pooling"
  bottom: "conv2/norm2"
  top: "pool2/3x3_s2"
  pooling_param {
    engine: MKL2017
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "inception_3a/1x1"
  type: "Convolution"
  bottom: "pool2/3x3_s2"
  top: "inception_3a/1x1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 64
    kernel_size: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_3a/relu_1x1"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_3a/1x1"
  top: "inception_3a/1x1"
}
layer {
  name: "inception_3a/3x3_reduce"
  type: "Convolution"
  bottom: "pool2/3x3_s2"
  top: "inception_3a/3x3_reduce"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 96
    kernel_size: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_3a/relu_3x3_reduce"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_3a/3x3_reduce"
  top: "inception_3a/3x3_reduce"
}
layer {
  name: "inception_3a/3x3"
  type: "Convolution"
  bottom: "inception_3a/3x3_reduce"
  top: "inception_3a/3x3"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 128
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_3a/relu_3x3"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_3a/3x3"
  top: "inception_3a/3x3"
}
layer {
  name: "inception_3a/5x5_reduce"
  type: "Convolution"
  bottom: "pool2/3x3_s2"
  top: "inception_3a/5x5_reduce"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 16
    kernel_size: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_3a/relu_5x5_reduce"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_3a/5x5_reduce"
  top: "inception_3a/5x5_reduce"
}
layer {
  name: "inception_3a/5x5"
  type: "Convolution"
  bottom: "inception_3a/5x5_reduce"
  top: "inception_3a/5x5"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 32
    pad: 2
    kernel_size: 5
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_3a/relu_5x5"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_3a/5x5"
  top: "inception_3a/5x5"
}
layer {
  name: "inception_3a/pool"
  type: "Pooling"
  bottom: "pool2/3x3_s2"
  top: "inception_3a/pool"
  pooling_param {
    engine: MKL2017
    pool: MAX
    kernel_size: 3
    stride: 1
    pad: 1
  }
}
layer {
  name: "inception_3a/pool_proj"
  type: "Convolution"
  bottom: "inception_3a/pool"
  top: "inception_3a/pool_proj"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 32
    kernel_size: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_3a/relu_pool_proj"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_3a/pool_proj"
  top: "inception_3a/pool_proj"
}
layer {
  name: "inception_3a/output"
  type: "Concat"
  concat_param {
    engine: MKL2017
  }
  bottom: "inception_3a/1x1"
  bottom: "inception_3a/3x3"
  bottom: "inception_3a/5x5"
  bottom: "inception_3a/pool_proj"
  top: "inception_3a/output"
}
layer {
  name: "inception_3b/1x1"
  type: "Convolution"
  bottom: "inception_3a/output"
  top: "inception_3b/1x1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 128
    kernel_size: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_3b/relu_1x1"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_3b/1x1"
  top: "inception_3b/1x1"
}
layer {
  name: "inception_3b/3x3_reduce"
  type: "Convolution"
  bottom: "inception_3a/output"
  top: "inception_3b/3x3_reduce"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 128
    kernel_size: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_3b/relu_3x3_reduce"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_3b/3x3_reduce"
  top: "inception_3b/3x3_reduce"
}
layer {
  name: "inception_3b/3x3"
  type: "Convolution"
  bottom: "inception_3b/3x3_reduce"
  top: "inception_3b/3x3"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 192
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_3b/relu_3x3"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_3b/3x3"
  top: "inception_3b/3x3"
}
layer {
  name: "inception_3b/5x5_reduce"
  type: "Convolution"
  bottom: "inception_3a/output"
  top: "inception_3b/5x5_reduce"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 32
    kernel_size: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_3b/relu_5x5_reduce"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_3b/5x5_reduce"
  top: "inception_3b/5x5_reduce"
}
layer {
  name: "inception_3b/5x5"
  type: "Convolution"
  bottom: "inception_3b/5x5_reduce"
  top: "inception_3b/5x5"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 96
    pad: 2
    kernel_size: 5
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_3b/relu_5x5"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_3b/5x5"
  top: "inception_3b/5x5"
}
layer {
  name: "inception_3b/pool"
  type: "Pooling"
  bottom: "inception_3a/output"
  top: "inception_3b/pool"
  pooling_param {
    engine: MKL2017
    pool: MAX
    kernel_size: 3
    stride: 1
    pad: 1
  }
}
layer {
  name: "inception_3b/pool_proj"
  type: "Convolution"
  bottom: "inception_3b/pool"
  top: "inception_3b/pool_proj"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 64
    kernel_size: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_3b/relu_pool_proj"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_3b/pool_proj"
  top: "inception_3b/pool_proj"
}
layer {
  name: "inception_3b/output"
  type: "Concat"
  concat_param {
    engine: MKL2017
  }
  bottom: "inception_3b/1x1"
  bottom: "inception_3b/3x3"
  bottom: "inception_3b/5x5"
  bottom: "inception_3b/pool_proj"
  top: "inception_3b/output"
}
layer {
  name: "pool3/3x3_s2"
  type: "Pooling"
  bottom: "inception_3b/output"
  top: "pool3/3x3_s2"
  pooling_param {
    engine: MKL2017
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "inception_4a/1x1"
  type: "Convolution"
  bottom: "pool3/3x3_s2"
  top: "inception_4a/1x1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 192
    kernel_size: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_4a/relu_1x1"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_4a/1x1"
  top: "inception_4a/1x1"
}
layer {
  name: "inception_4a/3x3_reduce"
  type: "Convolution"
  bottom: "pool3/3x3_s2"
  top: "inception_4a/3x3_reduce"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 96
    kernel_size: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_4a/relu_3x3_reduce"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_4a/3x3_reduce"
  top: "inception_4a/3x3_reduce"
}
layer {
  name: "inception_4a/3x3"
  type: "Convolution"
  bottom: "inception_4a/3x3_reduce"
  top: "inception_4a/3x3"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 208
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_4a/relu_3x3"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_4a/3x3"
  top: "inception_4a/3x3"
}
layer {
  name: "inception_4a/5x5_reduce"
  type: "Convolution"
  bottom: "pool3/3x3_s2"
  top: "inception_4a/5x5_reduce"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 16
    kernel_size: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_4a/relu_5x5_reduce"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_4a/5x5_reduce"
  top: "inception_4a/5x5_reduce"
}
layer {
  name: "inception_4a/5x5"
  type: "Convolution"
  bottom: "inception_4a/5x5_reduce"
  top: "inception_4a/5x5"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 48
    pad: 2
    kernel_size: 5
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_4a/relu_5x5"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_4a/5x5"
  top: "inception_4a/5x5"
}
layer {
  name: "inception_4a/pool"
  type: "Pooling"
  bottom: "pool3/3x3_s2"
  top: "inception_4a/pool"
  pooling_param {
    engine: MKL2017
    pool: MAX
    kernel_size: 3
    stride: 1
    pad: 1
  }
}
layer {
  name: "inception_4a/pool_proj"
  type: "Convolution"
  bottom: "inception_4a/pool"
  top: "inception_4a/pool_proj"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 64
    kernel_size: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_4a/relu_pool_proj"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_4a/pool_proj"
  top: "inception_4a/pool_proj"
}
layer {
  name: "inception_4a/output"
  type: "Concat"
  concat_param {
    engine: MKL2017
  }
  bottom: "inception_4a/1x1"
  bottom: "inception_4a/3x3"
  bottom: "inception_4a/5x5"
  bottom: "inception_4a/pool_proj"
  top: "inception_4a/output"
}
layer {
  name: "inception_4a/split"
  type: "Split"
  split_param {
    engine: MKL2017
  }
  bottom: "inception_4a/output"
  top: "inception_4b/input"
  top: "loss1_input"
}
layer {
  name: "loss1/ave_pool"
  type: "Pooling"
  bottom: "loss1_input"
  top: "loss1/ave_pool"
  pooling_param {
    engine: MKL2017
    pool: AVE
    kernel_size: 5
    stride: 3
  }
}
layer {
  name: "loss1/conv"
  type: "Convolution"
  bottom: "loss1/ave_pool"
  top: "loss1/conv"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 128
    kernel_size: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "loss1/relu_conv"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "loss1/conv"
  top: "loss1/conv"
}
layer {
  name: "loss1/fc"
  type: "InnerProduct"
  bottom: "loss1/conv"
  top: "loss1/fc"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  inner_product_param {
    num_output: 1024
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "loss1/relu_fc"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "loss1/fc"
  top: "loss1/fc"
}
# layer {
#   name: "loss1/drop_fc"
#   type: "Dropout"
#   bottom: "loss1/fc"
#   top: "loss1/fc"
#   dropout_param {
#     dropout_ratio: 0.7
#   }
# }
layer {
  name: "loss1/classifier"
  type: "InnerProduct"
  bottom: "loss1/fc"
  top: "loss1/classifier"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  inner_product_param {
    num_output: 1000
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "loss1/loss"
  type: "SoftmaxWithLoss"
  bottom: "loss1/classifier"
  bottom: "label"
  top: "loss1/loss1"
#  loss_weight: 0.3
  loss_weight: 1
}
layer {
  name: "loss1/top-1"
  type: "Accuracy"
  bottom: "loss1/classifier"
  bottom: "label"
  top: "loss1/top-1"
  include {
    phase: TEST
  }
}
layer {
  name: "loss1/top-5"
  type: "Accuracy"
  bottom: "loss1/classifier"
  bottom: "label"
  top: "loss1/top-5"
  include {
    phase: TEST
  }
  accuracy_param {
    top_k: 5
  }
}
layer {
  name: "inception_4b/1x1"
  type: "Convolution"
  bottom: "inception_4b/input"
  top: "inception_4b/1x1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 160
    kernel_size: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_4b/relu_1x1"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_4b/1x1"
  top: "inception_4b/1x1"
}
layer {
  name: "inception_4b/3x3_reduce"
  type: "Convolution"
  bottom: "inception_4b/input"
  top: "inception_4b/3x3_reduce"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 112
    kernel_size: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_4b/relu_3x3_reduce"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_4b/3x3_reduce"
  top: "inception_4b/3x3_reduce"
}
layer {
  name: "inception_4b/3x3"
  type: "Convolution"
  bottom: "inception_4b/3x3_reduce"
  top: "inception_4b/3x3"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 224
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_4b/relu_3x3"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_4b/3x3"
  top: "inception_4b/3x3"
}
layer {
  name: "inception_4b/5x5_reduce"
  type: "Convolution"
  bottom: "inception_4b/input"
  top: "inception_4b/5x5_reduce"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 24
    kernel_size: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_4b/relu_5x5_reduce"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_4b/5x5_reduce"
  top: "inception_4b/5x5_reduce"
}
layer {
  name: "inception_4b/5x5"
  type: "Convolution"
  bottom: "inception_4b/5x5_reduce"
  top: "inception_4b/5x5"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 64
    pad: 2
    kernel_size: 5
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_4b/relu_5x5"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_4b/5x5"
  top: "inception_4b/5x5"
}
layer {
  name: "inception_4b/pool"
  type: "Pooling"
  bottom: "inception_4b/input"
  top: "inception_4b/pool"
  pooling_param {
    engine: MKL2017
    pool: MAX
    kernel_size: 3
    stride: 1
    pad: 1
  }
}
layer {
  name: "inception_4b/pool_proj"
  type: "Convolution"
  bottom: "inception_4b/pool"
  top: "inception_4b/pool_proj"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 64
    kernel_size: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_4b/relu_pool_proj"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_4b/pool_proj"
  top: "inception_4b/pool_proj"
}
layer {
  name: "inception_4b/output"
  type: "Concat"
  concat_param {
    engine: MKL2017
  }
  bottom: "inception_4b/1x1"
  bottom: "inception_4b/3x3"
  bottom: "inception_4b/5x5"
  bottom: "inception_4b/pool_proj"
  top: "inception_4b/output"
}
layer {
  name: "inception_4c/1x1"
  type: "Convolution"
  bottom: "inception_4b/output"
  top: "inception_4c/1x1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 128
    kernel_size: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_4c/relu_1x1"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_4c/1x1"
  top: "inception_4c/1x1"
}
layer {
  name: "inception_4c/3x3_reduce"
  type: "Convolution"
  bottom: "inception_4b/output"
  top: "inception_4c/3x3_reduce"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }

  convolution_param {
    engine: MKL2017
    num_output: 128
    kernel_size: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_4c/relu_3x3_reduce"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_4c/3x3_reduce"
  top: "inception_4c/3x3_reduce"
}
layer {
  name: "inception_4c/3x3"
  type: "Convolution"
  bottom: "inception_4c/3x3_reduce"
  top: "inception_4c/3x3"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 256
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_4c/relu_3x3"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_4c/3x3"
  top: "inception_4c/3x3"
}
layer {
  name: "inception_4c/5x5_reduce"
  type: "Convolution"
  bottom: "inception_4b/output"
  top: "inception_4c/5x5_reduce"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 24
    kernel_size: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_4c/relu_5x5_reduce"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_4c/5x5_reduce"
  top: "inception_4c/5x5_reduce"
}
layer {
  name: "inception_4c/5x5"
  type: "Convolution"
  bottom: "inception_4c/5x5_reduce"
  top: "inception_4c/5x5"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 64
    pad: 2
    kernel_size: 5
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_4c/relu_5x5"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_4c/5x5"
  top: "inception_4c/5x5"
}
layer {
  name: "inception_4c/pool"
  type: "Pooling"
  bottom: "inception_4b/output"
  top: "inception_4c/pool"
  pooling_param {
    engine: MKL2017
    pool: MAX
    kernel_size: 3
    stride: 1
    pad: 1
  }
}
layer {
  name: "inception_4c/pool_proj"
  type: "Convolution"
  bottom: "inception_4c/pool"
  top: "inception_4c/pool_proj"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 64
    kernel_size: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_4c/relu_pool_proj"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_4c/pool_proj"
  top: "inception_4c/pool_proj"
}
layer {
  name: "inception_4c/output"
  type: "Concat"
  concat_param {
    engine: MKL2017
  }
  bottom: "inception_4c/1x1"
  bottom: "inception_4c/3x3"
  bottom: "inception_4c/5x5"
  bottom: "inception_4c/pool_proj"
  top: "inception_4c/output"
}
layer {
  name: "inception_4d/1x1"
  type: "Convolution"
  bottom: "inception_4c/output"
  top: "inception_4d/1x1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 112
    kernel_size: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_4d/relu_1x1"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_4d/1x1"
  top: "inception_4d/1x1"
}
layer {
  name: "inception_4d/3x3_reduce"
  type: "Convolution"
  bottom: "inception_4c/output"
  top: "inception_4d/3x3_reduce"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 144
    kernel_size: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_4d/relu_3x3_reduce"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_4d/3x3_reduce"
  top: "inception_4d/3x3_reduce"
}
layer {
  name: "inception_4d/3x3"
  type: "Convolution"
  bottom: "inception_4d/3x3_reduce"
  top: "inception_4d/3x3"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 288
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_4d/relu_3x3"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_4d/3x3"
  top: "inception_4d/3x3"
}
layer {
  name: "inception_4d/5x5_reduce"
  type: "Convolution"
  bottom: "inception_4c/output"
  top: "inception_4d/5x5_reduce"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 32
    kernel_size: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_4d/relu_5x5_reduce"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_4d/5x5_reduce"
  top: "inception_4d/5x5_reduce"
}
layer {
  name: "inception_4d/5x5"
  type: "Convolution"
  bottom: "inception_4d/5x5_reduce"
  top: "inception_4d/5x5"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 64
    pad: 2
    kernel_size: 5
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_4d/relu_5x5"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_4d/5x5"
  top: "inception_4d/5x5"
}
layer {
  name: "inception_4d/pool"
  type: "Pooling"
  bottom: "inception_4c/output"
  top: "inception_4d/pool"
  pooling_param {
    engine: MKL2017
    pool: MAX
    kernel_size: 3
    stride: 1
    pad: 1
  }
}
layer {
  name: "inception_4d/pool_proj"
  type: "Convolution"
  bottom: "inception_4d/pool"
  top: "inception_4d/pool_proj"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 64
    kernel_size: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_4d/relu_pool_proj"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_4d/pool_proj"
  top: "inception_4d/pool_proj"
}
layer {
  name: "inception_4d/output"
  type: "Concat"
  concat_param {
    engine: MKL2017
  }
  bottom: "inception_4d/1x1"
  bottom: "inception_4d/3x3"
  bottom: "inception_4d/5x5"
  bottom: "inception_4d/pool_proj"
  top: "inception_4d/output"
}
layer {
  name: "inception_4d/split"
  type: "Split"
  split_param {
    engine: MKL2017
  }
  bottom: "inception_4d/output"
  top: "inception_4e/input"
  top: "loss2_input"
}
layer {
  name: "loss2/ave_pool"
  type: "Pooling"
  bottom: "loss2_input"
  top: "loss2/ave_pool"
  pooling_param {
    engine: MKL2017
    pool: AVE
    kernel_size: 5
    stride: 3
  }
}
layer {
  name: "loss2/conv"
  type: "Convolution"
  bottom: "loss2/ave_pool"
  top: "loss2/conv"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 128
    kernel_size: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "loss2/relu_conv"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "loss2/conv"
  top: "loss2/conv"
}
layer {
  name: "loss2/fc"
  type: "InnerProduct"
  bottom: "loss2/conv"
  top: "loss2/fc"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  inner_product_param {
    num_output: 1024
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "loss2/relu_fc"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "loss2/fc"
  top: "loss2/fc"
}
# layer {
#   name: "loss2/drop_fc"
#   type: "Dropout"
#   bottom: "loss2/fc"
#   top: "loss2/fc"
#   dropout_param {
#     dropout_ratio: 0.7
#   }
# }
layer {
  name: "loss2/classifier"
  type: "InnerProduct"
  bottom: "loss2/fc"
  top: "loss2/classifier"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  inner_product_param {
    num_output: 1000
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "loss2/loss"
  type: "SoftmaxWithLoss"
  bottom: "loss2/classifier"
  bottom: "label"
  top: "loss2/loss1"
#  loss_weight: 0.3
  loss_weight: 1
}
layer {
  name: "loss2/top-1"
  type: "Accuracy"
  bottom: "loss2/classifier"
  bottom: "label"
  top: "loss2/top-1"
  include {
    phase: TEST
  }
}
layer {
  name: "loss2/top-5"
  type: "Accuracy"
  bottom: "loss2/classifier"
  bottom: "label"
  top: "loss2/top-5"
  include {
    phase: TEST
  }
  accuracy_param {
    top_k: 5
  }
}
layer {
  name: "inception_4e/1x1"
  type: "Convolution"
  bottom: "inception_4e/input"
  top: "inception_4e/1x1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 256
    kernel_size: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_4e/relu_1x1"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_4e/1x1"
  top: "inception_4e/1x1"
}
layer {
  name: "inception_4e/3x3_reduce"
  type: "Convolution"
  bottom: "inception_4e/input"
  top: "inception_4e/3x3_reduce"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 160
    kernel_size: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_4e/relu_3x3_reduce"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_4e/3x3_reduce"
  top: "inception_4e/3x3_reduce"
}
layer {
  name: "inception_4e/3x3"
  type: "Convolution"
  bottom: "inception_4e/3x3_reduce"
  top: "inception_4e/3x3"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 320
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_4e/relu_3x3"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_4e/3x3"
  top: "inception_4e/3x3"
}
layer {
  name: "inception_4e/5x5_reduce"
  type: "Convolution"
  bottom: "inception_4e/input"
  top: "inception_4e/5x5_reduce"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 32
    kernel_size: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_4e/relu_5x5_reduce"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_4e/5x5_reduce"
  top: "inception_4e/5x5_reduce"
}
layer {
  name: "inception_4e/5x5"
  type: "Convolution"
  bottom: "inception_4e/5x5_reduce"
  top: "inception_4e/5x5"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 128
    pad: 2
    kernel_size: 5
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_4e/relu_5x5"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_4e/5x5"
  top: "inception_4e/5x5"
}
layer {
  name: "inception_4e/pool"
  type: "Pooling"
  bottom: "inception_4e/input"
  top: "inception_4e/pool"
  pooling_param {
    engine: MKL2017
    pool: MAX
    kernel_size: 3
    stride: 1
    pad: 1
  }
}
layer {
  name: "inception_4e/pool_proj"
  type: "Convolution"
  bottom: "inception_4e/pool"
  top: "inception_4e/pool_proj"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 128
    kernel_size: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_4e/relu_pool_proj"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_4e/pool_proj"
  top: "inception_4e/pool_proj"
}
layer {
  name: "inception_4e/output"
  type: "Concat"
  concat_param {
    engine: MKL2017
  }
  bottom: "inception_4e/1x1"
  bottom: "inception_4e/3x3"
  bottom: "inception_4e/5x5"
  bottom: "inception_4e/pool_proj"
  top: "inception_4e/output"
}
layer {
  name: "pool4/3x3_s2"
  type: "Pooling"
  bottom: "inception_4e/output"
  top: "pool4/3x3_s2"
  pooling_param {
    engine: MKL2017
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "inception_5a/1x1"
  type: "Convolution"
  bottom: "pool4/3x3_s2"
  top: "inception_5a/1x1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 256
    kernel_size: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_5a/relu_1x1"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_5a/1x1"
  top: "inception_5a/1x1"
}
layer {
  name: "inception_5a/3x3_reduce"
  type: "Convolution"
  bottom: "pool4/3x3_s2"
  top: "inception_5a/3x3_reduce"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 160
    kernel_size: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_5a/relu_3x3_reduce"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_5a/3x3_reduce"
  top: "inception_5a/3x3_reduce"
}
layer {
  name: "inception_5a/3x3"
  type: "Convolution"
  bottom: "inception_5a/3x3_reduce"
  top: "inception_5a/3x3"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 320
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_5a/relu_3x3"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_5a/3x3"
  top: "inception_5a/3x3"
}
layer {
  name: "inception_5a/5x5_reduce"
  type: "Convolution"
  bottom: "pool4/3x3_s2"
  top: "inception_5a/5x5_reduce"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 32
    kernel_size: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_5a/relu_5x5_reduce"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_5a/5x5_reduce"
  top: "inception_5a/5x5_reduce"
}
layer {
  name: "inception_5a/5x5"
  type: "Convolution"
  bottom: "inception_5a/5x5_reduce"
  top: "inception_5a/5x5"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 128
    pad: 2
    kernel_size: 5
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_5a/relu_5x5"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_5a/5x5"
  top: "inception_5a/5x5"
}
layer {
  name: "inception_5a/pool"
  type: "Pooling"
  bottom: "pool4/3x3_s2"
  top: "inception_5a/pool"
  pooling_param {
    engine: MKL2017
    pool: MAX
    kernel_size: 3
    stride: 1
    pad: 1
  }
}
layer {
  name: "inception_5a/pool_proj"
  type: "Convolution"
  bottom: "inception_5a/pool"
  top: "inception_5a/pool_proj"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 128
    kernel_size: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_5a/relu_pool_proj"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_5a/pool_proj"
  top: "inception_5a/pool_proj"
}
layer {
  name: "inception_5a/output"
  type: "Concat"
  concat_param {
    engine: MKL2017
  }
  bottom: "inception_5a/1x1"
  bottom: "inception_5a/3x3"
  bottom: "inception_5a/5x5"
  bottom: "inception_5a/pool_proj"
  top: "inception_5a/output"
}
layer {
  name: "inception_5b/1x1"
  type: "Convolution"
  bottom: "inception_5a/output"
  top: "inception_5b/1x1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 384
    kernel_size: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_5b/relu_1x1"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_5b/1x1"
  top: "inception_5b/1x1"
}
layer {
  name: "inception_5b/3x3_reduce"
  type: "Convolution"
  bottom: "inception_5a/output"
  top: "inception_5b/3x3_reduce"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 192
    kernel_size: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_5b/relu_3x3_reduce"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_5b/3x3_reduce"
  top: "inception_5b/3x3_reduce"
}
layer {
  name: "inception_5b/3x3"
  type: "Convolution"
  bottom: "inception_5b/3x3_reduce"
  top: "inception_5b/3x3"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 384
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_5b/relu_3x3"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_5b/3x3"
  top: "inception_5b/3x3"
}
layer {
  name: "inception_5b/5x5_reduce"
  type: "Convolution"
  bottom: "inception_5a/output"
  top: "inception_5b/5x5_reduce"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 48
    kernel_size: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_5b/relu_5x5_reduce"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_5b/5x5_reduce"
  top: "inception_5b/5x5_reduce"
}
layer {
  name: "inception_5b/5x5"
  type: "Convolution"
  bottom: "inception_5b/5x5_reduce"
  top: "inception_5b/5x5"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 128
    pad: 2
    kernel_size: 5
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_5b/relu_5x5"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_5b/5x5"
  top: "inception_5b/5x5"
}
layer {
  name: "inception_5b/pool"
  type: "Pooling"
  bottom: "inception_5a/output"
  top: "inception_5b/pool"
  pooling_param {
    engine: MKL2017
    pool: MAX
    kernel_size: 3
    stride: 1
    pad: 1
  }
}
layer {
  name: "inception_5b/pool_proj"
  type: "Convolution"
  bottom: "inception_5b/pool"
  top: "inception_5b/pool_proj"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    engine: MKL2017
    num_output: 128
    kernel_size: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}
layer {
  name: "inception_5b/relu_pool_proj"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
  bottom: "inception_5b/pool_proj"
  top: "inception_5b/pool_proj"
}
layer {
  name: "inception_5b/output"
  type: "Concat"
  concat_param {
    engine: MKL2017
  }
  bottom: "inception_5b/1x1"
  bottom: "inception_5b/3x3"
  bottom: "inception_5b/5x5"
  bottom: "inception_5b/pool_proj"
  top: "inception_5b/output"
}
layer {
  name: "pool5/7x7_s1"
  type: "Pooling"
  bottom: "inception_5b/output"
  top: "pool5/7x7_s1"
  pooling_param {
    engine: MKL2017
    pool: AVE
    kernel_size: 7
    stride: 1
  }
}
# layer {
#   name: "pool5/drop_7x7_s1"
#   type: "Dropout"
#   bottom: "pool5/7x7_s1"
#   top: "pool5/7x7_s1"
#   dropout_param {
#     dropout_ratio: 0.4
#   }
# }
layer {
  name: "loss3/classifier"
  type: "InnerProduct"
  bottom: "pool5/7x7_s1"
  top: "loss3/classifier"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  inner_product_param {
    num_output: 1000
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "loss3/loss"
  type: "SoftmaxWithLoss"
  bottom: "loss3/classifier"
  bottom: "label"
  top: "loss3/loss"
  loss_weight: 1
}
layer {
  name: "loss3/top-1"
  type: "Accuracy"
  bottom: "loss3/classifier"
  bottom: "label"
  top: "loss3/top-1"
  include {
    phase: TEST
  }
}
layer {
  name: "loss3/top-5"
  type: "Accuracy"
  bottom: "loss3/classifier"
  bottom: "label"
  top: "loss3/top-5"
  include {
    phase: TEST
  }
  accuracy_param {
    top_k: 5
  }
}
"""
}

object GoogleNet_v1 {
  private def inception(inputSize: Int, config: Table, namePrefix: String): Module[Float] = {
    val concat = new Concat(2)
    val conv1 = new Sequential()
    val conv3 = new Sequential()
    val conv5 = new Sequential()
    val pool = new Sequential()

    conv1.add(new SpatialConvolution(inputSize, config[Table](1)(1), 1, 1, 1, 1)
      .setInitMethod(Xavier)
      .setName(namePrefix + "1x1"))
    conv1.add(new ReLU(ip = false).setName(namePrefix + "relu_1x1"))
    concat.add(conv1)

    conv3.add(new SpatialConvolution(inputSize, config[Table](2)(1), 1, 1, 1, 1)
      .setInitMethod(Xavier)
      .setName(namePrefix + "3x3_reduce"))
    conv3.add(new ReLU(ip = false).setName(namePrefix + "relu_3x3_reduce"))
    conv3.add(new SpatialConvolution(config[Table](2)(1), config[Table](2)(2), 3, 3, 1, 1, 1, 1)
      .setInitMethod(Xavier)
      .setName(namePrefix + "3x3"))
    conv3.add(new ReLU(ip = false).setName(namePrefix + "relu_3x3"))
    concat.add(conv3)

    conv5.add(new SpatialConvolution(inputSize, config[Table](3)(1), 1, 1, 1, 1)
      .setInitMethod(Xavier)
      .setName(namePrefix + "5x5_reduce"))
    conv5.add(new ReLU(ip = false).setName(namePrefix + "relu_5x5_reduce"))
    conv5.add(new SpatialConvolution(config[Table](3)(1), config[Table](3)(2), 5, 5, 1, 1, 2, 2)
      .setInitMethod(Xavier)
      .setName(namePrefix + "5x5"))
    conv5.add(new ReLU(ip = false).setName(namePrefix + "relu_5x5"))
    concat.add(conv5)

    pool.add(new SpatialMaxPooling(3, 3, 1, 1, 1, 1).ceil().setName(namePrefix + "pool"))
    pool.add(new SpatialConvolution(inputSize, config[Table](4)(1), 1, 1, 1, 1)
      .setInitMethod(Xavier)
      .setName(namePrefix + "pool_proj"))
    pool.add(new ReLU(ip = false).setName(namePrefix + "relu_pool_proj"))
    concat.add(pool).setName(namePrefix + "output")

    concat
  }

  def apply(classNum: Int): Module[Float] = {
    val feature1 = new Sequential()
    feature1.add(new SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3, propagateBack = true)
      .setInitMethod(Xavier)
      .setName("conv1/7x7_s2"))
    feature1.add(new ReLU(ip = false).setName("conv1/relu_7x7"))
    feature1.add(new SpatialMaxPooling(3, 3, 2, 2).ceil().setName("pool1/3x3_s2"))
    feature1.add(new SpatialCrossMapLRN(5, 0.0001, 0.75).setName("pool1/norm1"))
    feature1.add(new SpatialConvolution(64, 64, 1, 1, 1, 1).setInitMethod(Xavier)
      .setName("conv2/3x3_reduce"))
    feature1.add(new ReLU(ip = false).setName("conv2/relu_3x3_reduce"))
    feature1.add(new SpatialConvolution(64, 192, 3, 3, 1, 1, 1, 1).setInitMethod(Xavier)
      .setName("conv2/3x3"))
    feature1.add(new ReLU(ip = false).setName("conv2/relu_3x3"))
    feature1.add(new SpatialCrossMapLRN(5, 0.0001, 0.75).setName("conv2/norm2"))
    feature1.add(new SpatialMaxPooling(3, 3, 2, 2).ceil().setName("pool2/3x3_s2"))
    feature1.add(inception(192, T(T(64), T(96, 128), T(16, 32), T(32)), "inception_3a/"))
    feature1.add(inception(256, T(T(128), T(128, 192), T(32, 96), T(64)), "inception_3b/"))
    feature1.add(new SpatialMaxPooling(3, 3, 2, 2).ceil().setName("pool3/3x3_s2"))
    feature1.add(inception(480, T(T(192), T(96, 208), T(16, 48), T(64)), "inception_4a/"))

    val output1 = new Sequential()
    output1.add(new SpatialAveragePooling(5, 5, 3, 3).ceil().setName("loss1/ave_pool"))
    output1.add(new SpatialConvolution(512, 128, 1, 1, 1, 1).setName("loss1/conv"))
    output1.add(new ReLU(ip = false).setName("loss1/relu_conv"))
    output1.add(new View(128 * 4 * 4).setNumInputDims(3))
    output1.add(new Linear(128 * 4 * 4, 1024).setName("loss1/fc"))
    output1.add(new ReLU(ip = false).setName("loss1/relu_fc"))
    output1.add(new Dropout(0.7).setName("loss1/drop_fc"))
    output1.add(new Linear(1024, classNum).setName("loss1/classifier"))
    //    output1.add(LogSoftMax.setName("loss1/loss"))

    val feature2 = new Sequential()
    feature2.add(inception(512, T(T(160), T(112, 224), T(24, 64), T(64)), "inception_4b/"))
    feature2.add(inception(512, T(T(128), T(128, 256), T(24, 64), T(64)), "inception_4c/"))
    feature2.add(inception(512, T(T(112), T(144, 288), T(32, 64), T(64)), "inception_4d/"))

    val output2 = new Sequential()
    output2.add(new SpatialAveragePooling(5, 5, 3, 3).setName("loss2/ave_pool"))
    output2.add(new SpatialConvolution(528, 128, 1, 1, 1, 1).setName("loss2/conv"))
    output2.add(new ReLU(ip = false).setName("loss2/relu_conv"))
    output2.add(new View(128 * 4 * 4).setNumInputDims(3))
    output2.add(new Linear(128 * 4 * 4, 1024).setName("loss2/fc"))
    output2.add(new ReLU(ip = false).setName("loss2/relu_fc"))
    output2.add(new Dropout(0.7).setName("loss2/drop_fc"))
    output2.add(new Linear(1024, classNum).setName("loss2/classifier"))
    //    output2.add(LogSoftMax.setName("loss2/loss"))

    val output3 = new Sequential()
    output3.add(inception(528, T(T(256), T(160, 320), T(32, 128), T(128)), "inception_4e/"))
    output3.add(new SpatialMaxPooling(3, 3, 2, 2).ceil().setName("pool4/3x3_s2"))
    output3.add(inception(832, T(T(256), T(160, 320), T(32, 128), T(128)), "inception_5a/"))
    output3.add(inception(832, T(T(384), T(192, 384), T(48, 128), T(128)), "inception_5b/"))
    output3.add(new SpatialAveragePooling(7, 7, 1, 1).setName("pool5/7x7_s1"))
    output3.add(new Dropout(0.4).setName("pool5/drop_7x7_s1"))
    output3.add(new View(1024).setNumInputDims(3))
    output3.add(new Linear(1024, classNum).setInitMethod(Xavier).setName("loss3/classifier"))
    //    output3.add(LogSoftMax.setName("loss3/loss3"))

    val split2 = new Concat(2)
    split2.add(output3)
    split2.add(output2)

    val mainBranch = new Sequential()
    mainBranch.add(feature2)
    mainBranch.add(split2)

    val split1 = new Concat(2)
    split1.add(mainBranch)
    split1.add(output1)

    val model = new Sequential()

    model.add(feature1)
    model.add(split1)

    model.reset()
    model
  }
}

