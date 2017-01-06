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

import scala.collection.mutable.ArrayBuffer
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class GoogLeNetV2Spec extends FlatSpec with BeforeAndAfter with Matchers {
  before {
    Affinity.acquireCore()
  }

  "GoogLeNet generete output and gradient" should "correctly" in {
    val model = GoogleNet_v2(1000)
    model.convertToMklDnn()
    model.reset()

    val modules = ArrayBuffer[TensorModule[Float]]()
    Tools.flattenModules(model, modules)


    def iteration(): Unit = {
      val identity = Collect.run(googlenet_v2)
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

      val loss1Output = last1.output.asInstanceOf[Tensor[Float]]
      val loss2Output = last2.output.asInstanceOf[Tensor[Float]]
      val loss3Output = last3.output.asInstanceOf[Tensor[Float]]
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
  val googlenet_v2 =
    s"""
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
 bottom: "data"
  top: "conv1/7x7_s2"
  name: "conv1/7x7_s2"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
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
    bias_term: true
  }
}
layer {
  bottom: "conv1/7x7_s2"
  name: "conv1/7x7_s2/bn"
  top: "conv1/7x7_s2/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "conv1/7x7_s2/bn"
  top: "conv1/7x7_s2/bn"
  name: "conv1/7x7_s2/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
  bottom: "conv1/7x7_s2/bn"
  top: "pool1/3x3_s2"
  name: "pool1/3x3_s2"
  type: "Pooling"
  pooling_param {
    engine: MKL2017
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
 bottom: "pool1/3x3_s2"
  top: "conv2/3x3_reduce"
  name: "conv2/3x3_reduce"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 64
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "conv2/3x3_reduce"
  name: "conv2/3x3_reduce/bn"
  top: "conv2/3x3_reduce/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "conv2/3x3_reduce/bn"
  top: "conv2/3x3_reduce/bn"
  name: "conv2/3x3_reduce/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
 bottom: "conv2/3x3_reduce/bn"
  top: "conv2/3x3"
  name: "conv2/3x3"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 192
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "conv2/3x3"
  name: "conv2/3x3/bn"
  top: "conv2/3x3/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "conv2/3x3/bn"
  top: "conv2/3x3/bn"
  name: "conv2/3x3/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
  bottom: "conv2/3x3/bn"
  top: "pool2/3x3_s2"
  name: "pool2/3x3_s2"
  type: "Pooling"
  pooling_param {
    engine: MKL2017
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
 bottom: "pool2/3x3_s2"
  top: "inception_3a/1x1"
  name: "inception_3a/1x1"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 64
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_3a/1x1"
  name: "inception_3a/1x1/bn"
  top: "inception_3a/1x1/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_3a/1x1/bn"
  top: "inception_3a/1x1/bn"
  name: "inception_3a/1x1/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
 bottom: "pool2/3x3_s2"
  top: "inception_3a/3x3_reduce"
  name: "inception_3a/3x3_reduce"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 64
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_3a/3x3_reduce"
  name: "inception_3a/3x3_reduce/bn"
  top: "inception_3a/3x3_reduce/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_3a/3x3_reduce/bn"
  top: "inception_3a/3x3_reduce/bn"
  name: "inception_3a/3x3_reduce/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
 bottom: "inception_3a/3x3_reduce/bn"
  top: "inception_3a/3x3"
  name: "inception_3a/3x3"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 64
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_3a/3x3"
  name: "inception_3a/3x3/bn"
  top: "inception_3a/3x3/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_3a/3x3/bn"
  top: "inception_3a/3x3/bn"
  name: "inception_3a/3x3/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
 bottom: "pool2/3x3_s2"
  top: "inception_3a/double3x3_reduce"
  name: "inception_3a/double3x3_reduce"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 64
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_3a/double3x3_reduce"
  name: "inception_3a/double3x3_reduce/bn"
  top: "inception_3a/double3x3_reduce/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_3a/double3x3_reduce/bn"
  top: "inception_3a/double3x3_reduce/bn"
  name: "inception_3a/double3x3_reduce/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
 bottom: "inception_3a/double3x3_reduce/bn"
  top: "inception_3a/double3x3a"
  name: "inception_3a/double3x3a"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 96
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_3a/double3x3a"
  name: "inception_3a/double3x3a/bn"
  top: "inception_3a/double3x3a/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}

layer {
  bottom: "inception_3a/double3x3a/bn"
  top: "inception_3a/double3x3a/bn"
  name: "inception_3a/double3x3a/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
 bottom: "inception_3a/double3x3a/bn"
  top: "inception_3a/double3x3b"
  name: "inception_3a/double3x3b"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 96
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_3a/double3x3b"
  name: "inception_3a/double3x3b/bn"
  top: "inception_3a/double3x3b/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_3a/double3x3b/bn"
  top: "inception_3a/double3x3b/bn"
  name: "inception_3a/double3x3b/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
  bottom: "pool2/3x3_s2"
  top: "inception_3a/pool"
  name: "inception_3a/pool"
  type: "Pooling"
  pooling_param {
    engine: MKL2017
    pool: AVE
    kernel_size: 3
    stride: 1
    pad: 1
  }
}
layer {
 bottom: "inception_3a/pool"
  top: "inception_3a/pool_proj"
  name: "inception_3a/pool_proj"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 32
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_3a/pool_proj"
  name: "inception_3a/pool_proj/bn"
  top: "inception_3a/pool_proj/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_3a/pool_proj/bn"
  top: "inception_3a/pool_proj/bn"
  name: "inception_3a/pool_proj/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_3a/1x1/bn"
  bottom: "inception_3a/3x3/bn"
  bottom: "inception_3a/double3x3b/bn"
  bottom: "inception_3a/pool_proj/bn"
  top: "inception_3a/output"
  name: "inception_3a/output"
  type: "Concat"
  concat_param {
    engine: MKL2017
  }
}
layer {
 bottom: "inception_3a/output"
  top: "inception_3b/1x1"
  name: "inception_3b/1x1"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 64
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_3b/1x1"
  name: "inception_3b/1x1/bn"
  top: "inception_3b/1x1/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_3b/1x1/bn"
  top: "inception_3b/1x1/bn"
  name: "inception_3b/1x1/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
 bottom: "inception_3a/output"
  top: "inception_3b/3x3_reduce"
  name: "inception_3b/3x3_reduce"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 64
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_3b/3x3_reduce"
  name: "inception_3b/3x3_reduce/bn"
  top: "inception_3b/3x3_reduce/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}

layer {
  bottom: "inception_3b/3x3_reduce/bn"
  top: "inception_3b/3x3_reduce/bn"
  name: "inception_3b/3x3_reduce/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
 bottom: "inception_3b/3x3_reduce/bn"
  top: "inception_3b/3x3"
  name: "inception_3b/3x3"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 96
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_3b/3x3"
  name: "inception_3b/3x3/bn"
  top: "inception_3b/3x3/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_3b/3x3/bn"
  top: "inception_3b/3x3/bn"
  name: "inception_3b/3x3/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
 bottom: "inception_3a/output"
  top: "inception_3b/double3x3_reduce"
  name: "inception_3b/double3x3_reduce"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 64
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_3b/double3x3_reduce"
  name: "inception_3b/double3x3_reduce/bn"
  top: "inception_3b/double3x3_reduce/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_3b/double3x3_reduce/bn"
  top: "inception_3b/double3x3_reduce/bn"
  name: "inception_3b/double3x3_reduce/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
 bottom: "inception_3b/double3x3_reduce/bn"
  top: "inception_3b/double3x3a"
  name: "inception_3b/double3x3a"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 96
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_3b/double3x3a"
  name: "inception_3b/double3x3a/bn"
  top: "inception_3b/double3x3a/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_3b/double3x3a/bn"
  top: "inception_3b/double3x3a/bn"
  name: "inception_3b/double3x3a/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
 bottom: "inception_3b/double3x3a/bn"
  top: "inception_3b/double3x3b"
  name: "inception_3b/double3x3b"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 96
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_3b/double3x3b"
  name: "inception_3b/double3x3b/bn"
  top: "inception_3b/double3x3b/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_3b/double3x3b/bn"
  top: "inception_3b/double3x3b/bn"
  name: "inception_3b/double3x3b/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_3a/output"
  top: "inception_3b/pool"
  name: "inception_3b/pool"
  type: "Pooling"
  pooling_param {
    engine: MKL2017
    pool: AVE
    kernel_size: 3
    stride: 1
    pad: 1
  }
}
layer {
 bottom: "inception_3b/pool"
  top: "inception_3b/pool_proj"
  name: "inception_3b/pool_proj"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 64
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_3b/pool_proj"
  name: "inception_3b/pool_proj/bn"
  top: "inception_3b/pool_proj/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_3b/pool_proj/bn"
  top: "inception_3b/pool_proj/bn"
  name: "inception_3b/pool_proj/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_3b/1x1/bn"
  bottom: "inception_3b/3x3/bn"
  bottom: "inception_3b/double3x3b/bn"
  bottom: "inception_3b/pool_proj/bn"
  top: "inception_3b/output"
  name: "inception_3b/output"
  type: "Concat"
  concat_param {
    engine: MKL2017
  }
}
layer {
 bottom: "inception_3b/output"
  top: "inception_3c/3x3_reduce"
  name: "inception_3c/3x3_reduce"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 128
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_3c/3x3_reduce"
  name: "inception_3c/3x3_reduce/bn"
  top: "inception_3c/3x3_reduce/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_3c/3x3_reduce/bn"
  top: "inception_3c/3x3_reduce/bn"
  name: "inception_3c/3x3_reduce/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
 bottom: "inception_3c/3x3_reduce/bn"
  top: "inception_3c/3x3"
  name: "inception_3c/3x3"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 160
    pad: 1
    kernel_size: 3
    stride: 2
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_3c/3x3"
  name: "inception_3c/3x3/bn"
  top: "inception_3c/3x3/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_3c/3x3/bn"
  top: "inception_3c/3x3/bn"
  name: "inception_3c/3x3/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
 bottom: "inception_3b/output"
  top: "inception_3c/double3x3_reduce"
  name: "inception_3c/double3x3_reduce"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 64
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_3c/double3x3_reduce"
  name: "inception_3c/double3x3_reduce/bn"
  top: "inception_3c/double3x3_reduce/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_3c/double3x3_reduce/bn"
  top: "inception_3c/double3x3_reduce/bn"
  name: "inception_3c/double3x3_reduce/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
 bottom: "inception_3c/double3x3_reduce/bn"
  top: "inception_3c/double3x3a"
  name: "inception_3c/double3x3a"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 96
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_3c/double3x3a"
  name: "inception_3c/double3x3a/bn"
  top: "inception_3c/double3x3a/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}

layer {
  bottom: "inception_3c/double3x3a/bn"
  top: "inception_3c/double3x3a/bn"
  name: "inception_3c/double3x3a/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
 bottom: "inception_3c/double3x3a/bn"
  top: "inception_3c/double3x3b"
  name: "inception_3c/double3x3b"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 96
    pad: 1
    kernel_size: 3
    stride: 2
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_3c/double3x3b"
  name: "inception_3c/double3x3b/bn"
  top: "inception_3c/double3x3b/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_3c/double3x3b/bn"
  top: "inception_3c/double3x3b/bn"
  name: "inception_3c/double3x3b/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_3b/output"
  top: "inception_3c/pool"
  name: "inception_3c/pool"
  type: "Pooling"
  pooling_param {
    engine: MKL2017
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  bottom: "inception_3c/3x3/bn"
  bottom: "inception_3c/double3x3b/bn"
  bottom: "inception_3c/pool"
  top: "inception_3c/output"
  name: "inception_3c/output"
  type: "Concat"
  concat_param {
    engine: MKL2017
  }
}
layer {
  name: "inception_3c/split"
  type: "Split"
  split_param {
    engine: MKL2017
  }
  bottom: "inception_3c/output"
  top: "inception_4a/input"
  top: "loss1_input"
}
layer {
  bottom: "loss1_input"
  top: "pool3/5x5_s3"
  name: "pool3/5x5_s3"
  type: "Pooling"
  pooling_param {
    engine: MKL2017
    pool: AVE
    kernel_size: 5
    stride: 3
  }
}
layer {
 bottom: "pool3/5x5_s3"
  top: "loss1/conv"
  name: "loss1/conv"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 128
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "loss1/conv"
  name: "loss1/conv/bn"
  top: "loss1/conv/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "loss1/conv/bn"
  top: "loss1/conv/bn"
  name: "loss1/conv/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
  bottom: "loss1/conv/bn"
  top: "loss1/fc"
  name: "loss1/fc"
  type: "InnerProduct"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  inner_product_param {
    num_output: 1024
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
#layer {
#  bottom: "loss1/fc"
#  name: "loss1/fc/bn"
#  top: "loss1/fc/bn"
#  type: "BatchNorm"
#  batch_norm_param {
#    engine: MKL2017
#  }
#}
layer {
  bottom: "loss1/fc"
  top: "loss1/fc/bn"
  name: "loss1/fc/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
  bottom: "loss1/fc/bn"
  top: "loss1/classifier"
  name: "loss1/classifier"
  type: "InnerProduct"
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
  bottom: "loss1/classifier"
  bottom: "label"
  top: "loss1/loss"
  name: "loss1/loss"
  type: "SoftmaxWithLoss"
  loss_weight: 1
}
layer {
  bottom: "loss1/classifier"
  top: "loss1/prob"
  name: "loss1/prob"
  type: "Softmax"
  include {
    phase: TEST
  }
}
layer {
  bottom: "loss1/prob"
  bottom: "label"
  top: "loss1/top-1"
  name: "loss1/top-1"
  type: "Accuracy"
  include {
    phase: TEST
  }
}
layer {
  bottom: "loss1/prob"
  bottom: "label"
  top: "loss1/top-5"
  name: "loss1/top-5"
  type: "Accuracy"
  accuracy_param {
    top_k: 5
  }
  include {
    phase: TEST
  }
}
layer {
 bottom: "inception_4a/input"
  top: "inception_4a/1x1"
  name: "inception_4a/1x1"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 224
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_4a/1x1"
  name: "inception_4a/1x1/bn"
  top: "inception_4a/1x1/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_4a/1x1/bn"
  top: "inception_4a/1x1/bn"
  name: "inception_4a/1x1/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
 bottom: "inception_4a/input"
  top: "inception_4a/3x3_reduce"
  name: "inception_4a/3x3_reduce"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 64
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_4a/3x3_reduce"
  name: "inception_4a/3x3_reduce/bn"
  top: "inception_4a/3x3_reduce/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_4a/3x3_reduce/bn"
  top: "inception_4a/3x3_reduce/bn"
  name: "inception_4a/3x3_reduce/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
 bottom: "inception_4a/3x3_reduce/bn"
  top: "inception_4a/3x3"
  name: "inception_4a/3x3"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 96
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_4a/3x3"
  name: "inception_4a/3x3/bn"
  top: "inception_4a/3x3/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_4a/3x3/bn"
  top: "inception_4a/3x3/bn"
  name: "inception_4a/3x3/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
 bottom: "inception_4a/input"
  top: "inception_4a/double3x3_reduce"
  name: "inception_4a/double3x3_reduce"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 96
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_4a/double3x3_reduce"
  name: "inception_4a/double3x3_reduce/bn"
  top: "inception_4a/double3x3_reduce/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_4a/double3x3_reduce/bn"
  top: "inception_4a/double3x3_reduce/bn"
  name: "inception_4a/double3x3_reduce/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
 bottom: "inception_4a/double3x3_reduce/bn"
  top: "inception_4a/double3x3a"
  name: "inception_4a/double3x3a"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 128
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_4a/double3x3a"
  name: "inception_4a/double3x3a/bn"
  top: "inception_4a/double3x3a/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_4a/double3x3a/bn"
  top: "inception_4a/double3x3a/bn"
  name: "inception_4a/double3x3a/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
 bottom: "inception_4a/double3x3a/bn"
  top: "inception_4a/double3x3b"
  name: "inception_4a/double3x3b"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 128
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_4a/double3x3b"
  name: "inception_4a/double3x3b/bn"
  top: "inception_4a/double3x3b/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_4a/double3x3b/bn"
  top: "inception_4a/double3x3b/bn"
  name: "inception_4a/double3x3b/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_4a/input"
  top: "inception_4a/pool"
  name: "inception_4a/pool"
  type: "Pooling"
  pooling_param {
    engine: MKL2017
    pool: AVE
    kernel_size: 3
    stride: 1
    pad: 1
  }
}
layer {
 bottom: "inception_4a/pool"
  top: "inception_4a/pool_proj"
  name: "inception_4a/pool_proj"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 128
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_4a/pool_proj"
  name: "inception_4a/pool_proj/bn"
  top: "inception_4a/pool_proj/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_4a/pool_proj/bn"
  top: "inception_4a/pool_proj/bn"
  name: "inception_4a/pool_proj/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_4a/1x1/bn"
  bottom: "inception_4a/3x3/bn"
  bottom: "inception_4a/double3x3b/bn"
  bottom: "inception_4a/pool_proj/bn"
  top: "inception_4a/output"
  name: "inception_4a/output"
  type: "Concat"
  concat_param {
    engine: MKL2017
  }
}
layer {
 bottom: "inception_4a/output"
  top: "inception_4b/1x1"
  name: "inception_4b/1x1"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 192
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_4b/1x1"
  name: "inception_4b/1x1/bn"
  top: "inception_4b/1x1/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_4b/1x1/bn"
  top: "inception_4b/1x1/bn"
  name: "inception_4b/1x1/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
 bottom: "inception_4a/output"
  top: "inception_4b/3x3_reduce"
  name: "inception_4b/3x3_reduce"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 96
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_4b/3x3_reduce"
  name: "inception_4b/3x3_reduce/bn"
  top: "inception_4b/3x3_reduce/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_4b/3x3_reduce/bn"
  top: "inception_4b/3x3_reduce/bn"
  name: "inception_4b/3x3_reduce/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
 bottom: "inception_4b/3x3_reduce/bn"
  top: "inception_4b/3x3"
  name: "inception_4b/3x3"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 128
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_4b/3x3"
  name: "inception_4b/3x3/bn"
  top: "inception_4b/3x3/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_4b/3x3/bn"
  top: "inception_4b/3x3/bn"
  name: "inception_4b/3x3/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
 bottom: "inception_4a/output"
  top: "inception_4b/double3x3_reduce"
  name: "inception_4b/double3x3_reduce"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 96
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_4b/double3x3_reduce"
  name: "inception_4b/double3x3_reduce/bn"
  top: "inception_4b/double3x3_reduce/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_4b/double3x3_reduce/bn"
  top: "inception_4b/double3x3_reduce/bn"
  name: "inception_4b/double3x3_reduce/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
 bottom: "inception_4b/double3x3_reduce/bn"
  top: "inception_4b/double3x3a"
  name: "inception_4b/double3x3a"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 128
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_4b/double3x3a"
  name: "inception_4b/double3x3a/bn"
  top: "inception_4b/double3x3a/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_4b/double3x3a/bn"
  top: "inception_4b/double3x3a/bn"
  name: "inception_4b/double3x3a/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
 bottom: "inception_4b/double3x3a/bn"
  top: "inception_4b/double3x3b"
  name: "inception_4b/double3x3b"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 128
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_4b/double3x3b"
  name: "inception_4b/double3x3b/bn"
  top: "inception_4b/double3x3b/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_4b/double3x3b/bn"
  top: "inception_4b/double3x3b/bn"
  name: "inception_4b/double3x3b/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_4a/output"
  top: "inception_4b/pool"
  name: "inception_4b/pool"
  type: "Pooling"
  pooling_param {
    engine: MKL2017
    pool: AVE
    kernel_size: 3
    stride: 1
    pad: 1
  }
}
layer {
 bottom: "inception_4b/pool"
  top: "inception_4b/pool_proj"
  name: "inception_4b/pool_proj"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 128
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_4b/pool_proj"
  name: "inception_4b/pool_proj/bn"
  top: "inception_4b/pool_proj/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_4b/pool_proj/bn"
  top: "inception_4b/pool_proj/bn"
  name: "inception_4b/pool_proj/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_4b/1x1/bn"
  bottom: "inception_4b/3x3/bn"
  bottom: "inception_4b/double3x3b/bn"
  bottom: "inception_4b/pool_proj/bn"
  top: "inception_4b/output"
  name: "inception_4b/output"
  type: "Concat"
  concat_param {
    engine: MKL2017
  }
}
layer {
 bottom: "inception_4b/output"
  top: "inception_4c/1x1"
  name: "inception_4c/1x1"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 160
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_4c/1x1"
  name: "inception_4c/1x1/bn"
  top: "inception_4c/1x1/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_4c/1x1/bn"
  top: "inception_4c/1x1/bn"
  name: "inception_4c/1x1/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
 bottom: "inception_4b/output"
  top: "inception_4c/3x3_reduce"
  name: "inception_4c/3x3_reduce"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 128
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_4c/3x3_reduce"
  name: "inception_4c/3x3_reduce/bn"
  top: "inception_4c/3x3_reduce/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_4c/3x3_reduce/bn"
  top: "inception_4c/3x3_reduce/bn"
  name: "inception_4c/3x3_reduce/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
 bottom: "inception_4c/3x3_reduce/bn"
  top: "inception_4c/3x3"
  name: "inception_4c/3x3"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 160
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_4c/3x3"
  name: "inception_4c/3x3/bn"
  top: "inception_4c/3x3/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_4c/3x3/bn"
  top: "inception_4c/3x3/bn"
  name: "inception_4c/3x3/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
 bottom: "inception_4b/output"
  top: "inception_4c/double3x3_reduce"
  name: "inception_4c/double3x3_reduce"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 128
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_4c/double3x3_reduce"
  name: "inception_4c/double3x3_reduce/bn"
  top: "inception_4c/double3x3_reduce/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_4c/double3x3_reduce/bn"
  top: "inception_4c/double3x3_reduce/bn"
  name: "inception_4c/double3x3_reduce/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
 bottom: "inception_4c/double3x3_reduce/bn"
  top: "inception_4c/double3x3a"
  name: "inception_4c/double3x3a"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 160
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_4c/double3x3a"
  name: "inception_4c/double3x3a/bn"
  top: "inception_4c/double3x3a/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_4c/double3x3a/bn"
  top: "inception_4c/double3x3a/bn"
  name: "inception_4c/double3x3a/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
 bottom: "inception_4c/double3x3a/bn"
  top: "inception_4c/double3x3b"
  name: "inception_4c/double3x3b"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 160
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_4c/double3x3b"
  name: "inception_4c/double3x3b/bn"
  top: "inception_4c/double3x3b/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_4c/double3x3b/bn"
  top: "inception_4c/double3x3b/bn"
  name: "inception_4c/double3x3b/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_4b/output"
  top: "inception_4c/pool"
  name: "inception_4c/pool"
  type: "Pooling"
  pooling_param {
    engine: MKL2017
    pool: AVE
    kernel_size: 3
    stride: 1
    pad: 1
  }
}
layer {
 bottom: "inception_4c/pool"
  top: "inception_4c/pool_proj"
  name: "inception_4c/pool_proj"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 96
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_4c/pool_proj"
  name: "inception_4c/pool_proj/bn"
  top: "inception_4c/pool_proj/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_4c/pool_proj/bn"
  top: "inception_4c/pool_proj/bn"
  name: "inception_4c/pool_proj/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_4c/1x1/bn"
  bottom: "inception_4c/3x3/bn"
  bottom: "inception_4c/double3x3b/bn"
  bottom: "inception_4c/pool_proj/bn"
  top: "inception_4c/output"
  name: "inception_4c/output"
  type: "Concat"
  concat_param {
    engine: MKL2017
  }
}
layer {
 bottom: "inception_4c/output"
  top: "inception_4d/1x1"
  name: "inception_4d/1x1"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 96
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_4d/1x1"
  name: "inception_4d/1x1/bn"
  top: "inception_4d/1x1/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_4d/1x1/bn"
  top: "inception_4d/1x1/bn"
  name: "inception_4d/1x1/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
 bottom: "inception_4c/output"
  top: "inception_4d/3x3_reduce"
  name: "inception_4d/3x3_reduce"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 128
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_4d/3x3_reduce"
  name: "inception_4d/3x3_reduce/bn"
  top: "inception_4d/3x3_reduce/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_4d/3x3_reduce/bn"
  top: "inception_4d/3x3_reduce/bn"
  name: "inception_4d/3x3_reduce/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
 bottom: "inception_4d/3x3_reduce/bn"
  top: "inception_4d/3x3"
  name: "inception_4d/3x3"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 192
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_4d/3x3"
  name: "inception_4d/3x3/bn"
  top: "inception_4d/3x3/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_4d/3x3/bn"
  top: "inception_4d/3x3/bn"
  name: "inception_4d/3x3/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
 bottom: "inception_4c/output"
  top: "inception_4d/double3x3_reduce"
  name: "inception_4d/double3x3_reduce"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 160
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_4d/double3x3_reduce"
  name: "inception_4d/double3x3_reduce/bn"
  top: "inception_4d/double3x3_reduce/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_4d/double3x3_reduce/bn"
  top: "inception_4d/double3x3_reduce/bn"
  name: "inception_4d/double3x3_reduce/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
 bottom: "inception_4d/double3x3_reduce/bn"
  top: "inception_4d/double3x3a"
  name: "inception_4d/double3x3a"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 192
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_4d/double3x3a"
  name: "inception_4d/double3x3a/bn"
  top: "inception_4d/double3x3a/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_4d/double3x3a/bn"
  top: "inception_4d/double3x3a/bn"
  name: "inception_4d/double3x3a/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
 bottom: "inception_4d/double3x3a/bn"
  top: "inception_4d/double3x3b"
  name: "inception_4d/double3x3b"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 192
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_4d/double3x3b"
  name: "inception_4d/double3x3b/bn"
  top: "inception_4d/double3x3b/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_4d/double3x3b/bn"
  top: "inception_4d/double3x3b/bn"
  name: "inception_4d/double3x3b/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_4c/output"
  top: "inception_4d/pool"
  name: "inception_4d/pool"
  type: "Pooling"
  pooling_param {
    engine: MKL2017
    pool: AVE
    kernel_size: 3
    stride: 1
    pad: 1
  }
}
layer {
 bottom: "inception_4d/pool"
  top: "inception_4d/pool_proj"
  name: "inception_4d/pool_proj"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 96
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_4d/pool_proj"
  name: "inception_4d/pool_proj/bn"
  top: "inception_4d/pool_proj/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_4d/pool_proj/bn"
  top: "inception_4d/pool_proj/bn"
  name: "inception_4d/pool_proj/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_4d/1x1/bn"
  bottom: "inception_4d/3x3/bn"
  bottom: "inception_4d/double3x3b/bn"
  bottom: "inception_4d/pool_proj/bn"
  top: "inception_4d/output"
  name: "inception_4d/output"
  type: "Concat"
  concat_param {
    engine: MKL2017
  }
}
layer {
 bottom: "inception_4d/output"
  top: "inception_4e/3x3_reduce"
  name: "inception_4e/3x3_reduce"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 128
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_4e/3x3_reduce"
  name: "inception_4e/3x3_reduce/bn"
  top: "inception_4e/3x3_reduce/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_4e/3x3_reduce/bn"
  top: "inception_4e/3x3_reduce/bn"
  name: "inception_4e/3x3_reduce/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
 bottom: "inception_4e/3x3_reduce/bn"
  top: "inception_4e/3x3"
  name: "inception_4e/3x3"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 192
    pad: 1
    kernel_size: 3
    stride: 2
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_4e/3x3"
  name: "inception_4e/3x3/bn"
  top: "inception_4e/3x3/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_4e/3x3/bn"
  top: "inception_4e/3x3/bn"
  name: "inception_4e/3x3/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
 bottom: "inception_4d/output"
  top: "inception_4e/double3x3_reduce"
  name: "inception_4e/double3x3_reduce"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 192
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_4e/double3x3_reduce"
  name: "inception_4e/double3x3_reduce/bn"
  top: "inception_4e/double3x3_reduce/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_4e/double3x3_reduce/bn"
  top: "inception_4e/double3x3_reduce/bn"
  name: "inception_4e/double3x3_reduce/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
 bottom: "inception_4e/double3x3_reduce/bn"
  top: "inception_4e/double3x3a"
  name: "inception_4e/double3x3a"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 256
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_4e/double3x3a"
  name: "inception_4e/double3x3a/bn"
  top: "inception_4e/double3x3a/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_4e/double3x3a/bn"
  top: "inception_4e/double3x3a/bn"
  name: "inception_4e/double3x3a/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
 bottom: "inception_4e/double3x3a/bn"
  top: "inception_4e/double3x3b"
  name: "inception_4e/double3x3b"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 256
    pad: 1
    kernel_size: 3
    stride: 2
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_4e/double3x3b"
  name: "inception_4e/double3x3b/bn"
  top: "inception_4e/double3x3b/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_4e/double3x3b/bn"
  top: "inception_4e/double3x3b/bn"
  name: "inception_4e/double3x3b/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_4d/output"
  top: "inception_4e/pool"
  name: "inception_4e/pool"
  type: "Pooling"
  pooling_param {
    engine: MKL2017
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  bottom: "inception_4e/3x3/bn"
  bottom: "inception_4e/double3x3b/bn"
  bottom: "inception_4e/pool"
  top: "inception_4e/output"
  name: "inception_4e/output"
  type: "Concat"
  concat_param {
    engine: MKL2017
  }
}
layer {
  name: "inception_4e/split"
  type: "Split"
  split_param {
    engine: MKL2017
  }
  bottom: "inception_4e/output"
  top: "inception_5a/input"
  top: "loss2_input"
}
layer {
  bottom: "loss2_input"
  top: "pool4/5x5_s3"
  name: "pool4/5x5_s3"
  type: "Pooling"
  pooling_param {
    engine: MKL2017
    pool: AVE
    kernel_size: 5
    stride: 3
  }
}
layer {
 bottom: "pool4/5x5_s3"
  top: "loss2/conv"
  name: "loss2/conv"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 128
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "loss2/conv"
  name: "loss2/conv/bn"
  top: "loss2/conv/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "loss2/conv/bn"
  top: "loss2/conv/bn/relu"
  name: "loss2/conv/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
  bottom: "loss2/conv/bn/relu"
  top: "loss2/fc"
  name: "loss2/fc"
  type: "InnerProduct"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  inner_product_param {
    num_output: 1024
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
#layer {
#  bottom: "loss2/fc"
#  name: "loss2/fc/bn"
#  top: "loss2/fc/bn"
#  type: "BatchNorm"
#  batch_norm_param {
#    engine: MKL2017
#  }
#}
layer {
  bottom: "loss2/fc"
  top: "loss2/fc/bn/relu"
  name: "loss2/fc/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
  bottom: "loss2/fc/bn/relu"
  top: "loss2/classifier"
  name: "loss2/classifier"
  type: "InnerProduct"
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
  bottom: "loss2/classifier"
  bottom: "label"
  top: "loss2/loss"
  name: "loss2/loss"
  type: "SoftmaxWithLoss"
  loss_weight: 1
}
layer {
  bottom: "loss2/classifier"
  top: "loss2/prob"
  name: "loss2/prob"
  type: "Softmax"
  include {
    phase: TEST
  }
}
layer {
  bottom: "loss2/prob"
  bottom: "label"
  top: "loss2/top-1"
  name: "loss2/top-1"
  type: "Accuracy"
  include {
    phase: TEST
  }
}
layer {
  bottom: "loss2/prob"
  bottom: "label"
  top: "loss2/top-5"
  name: "loss2/top-5"
  type: "Accuracy"
  accuracy_param {
    top_k: 5
  }
  include {
    phase: TEST
  }
}
layer {
 bottom: "inception_5a/input"
  top: "inception_5a/1x1"
  name: "inception_5a/1x1"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 352
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_5a/1x1"
  name: "inception_5a/1x1/bn"
  top: "inception_5a/1x1/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_5a/1x1/bn"
  top: "inception_5a/1x1/bn"
  name: "inception_5a/1x1/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
 bottom: "inception_5a/input"
  top: "inception_5a/3x3_reduce"
  name: "inception_5a/3x3_reduce"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 192
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_5a/3x3_reduce"
  name: "inception_5a/3x3_reduce/bn"
  top: "inception_5a/3x3_reduce/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_5a/3x3_reduce/bn"
  top: "inception_5a/3x3_reduce/bn"
  name: "inception_5a/3x3_reduce/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
 bottom: "inception_5a/3x3_reduce/bn"
  top: "inception_5a/3x3"
  name: "inception_5a/3x3"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 320
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_5a/3x3"
  name: "inception_5a/3x3/bn"
  top: "inception_5a/3x3/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_5a/3x3/bn"
  top: "inception_5a/3x3/bn"
  name: "inception_5a/3x3/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
 bottom: "inception_5a/input"
  top: "inception_5a/double3x3_reduce"
  name: "inception_5a/double3x3_reduce"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 160
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_5a/double3x3_reduce"
  name: "inception_5a/double3x3_reduce/bn"
  top: "inception_5a/double3x3_reduce/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_5a/double3x3_reduce/bn"
  top: "inception_5a/double3x3_reduce/bn"
  name: "inception_5a/double3x3_reduce/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
 bottom: "inception_5a/double3x3_reduce/bn"
  top: "inception_5a/double3x3a"
  name: "inception_5a/double3x3a"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 224
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_5a/double3x3a"
  name: "inception_5a/double3x3a/bn"
  top: "inception_5a/double3x3a/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_5a/double3x3a/bn"
  top: "inception_5a/double3x3a/bn"
  name: "inception_5a/double3x3a/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
 bottom: "inception_5a/double3x3a/bn"
  top: "inception_5a/double3x3b"
  name: "inception_5a/double3x3b"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 224
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_5a/double3x3b"
  name: "inception_5a/double3x3b/bn"
  top: "inception_5a/double3x3b/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_5a/double3x3b/bn"
  top: "inception_5a/double3x3b/bn"
  name: "inception_5a/double3x3b/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_5a/input"
  top: "inception_5a/pool"
  name: "inception_5a/pool"
  type: "Pooling"
  pooling_param {
    engine: MKL2017
    pool: AVE
    kernel_size: 3
    stride: 1
    pad: 1
  }
}
layer {
 bottom: "inception_5a/pool"
  top: "inception_5a/pool_proj"
  name: "inception_5a/pool_proj"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 128
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_5a/pool_proj"
  name: "inception_5a/pool_proj/bn"
  top: "inception_5a/pool_proj/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_5a/pool_proj/bn"
  top: "inception_5a/pool_proj/bn"
  name: "inception_5a/pool_proj/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_5a/1x1/bn"
  bottom: "inception_5a/3x3/bn"
  bottom: "inception_5a/double3x3b/bn"
  bottom: "inception_5a/pool_proj/bn"
  top: "inception_5a/output"
  name: "inception_5a/output"
  type: "Concat"
  concat_param {
    engine: MKL2017
  }
}
layer {
 bottom: "inception_5a/output"
  top: "inception_5b/1x1"
  name: "inception_5b/1x1"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 352
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_5b/1x1"
  name: "inception_5b/1x1/bn"
  top: "inception_5b/1x1/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_5b/1x1/bn"
  top: "inception_5b/1x1/bn"
  name: "inception_5b/1x1/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
 bottom: "inception_5a/output"
  top: "inception_5b/3x3_reduce"
  name: "inception_5b/3x3_reduce"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 192
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_5b/3x3_reduce"
  name: "inception_5b/3x3_reduce/bn"
  top: "inception_5b/3x3_reduce/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_5b/3x3_reduce/bn"
  top: "inception_5b/3x3_reduce/bn"
  name: "inception_5b/3x3_reduce/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
 bottom: "inception_5b/3x3_reduce/bn"
  top: "inception_5b/3x3"
  name: "inception_5b/3x3"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 320
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_5b/3x3"
  name: "inception_5b/3x3/bn"
  top: "inception_5b/3x3/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_5b/3x3/bn"
  top: "inception_5b/3x3/bn"
  name: "inception_5b/3x3/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
 bottom: "inception_5a/output"
  top: "inception_5b/double3x3_reduce"
  name: "inception_5b/double3x3_reduce"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 192
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_5b/double3x3_reduce"
  name: "inception_5b/double3x3_reduce/bn"
  top: "inception_5b/double3x3_reduce/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_5b/double3x3_reduce/bn"
  top: "inception_5b/double3x3_reduce/bn"
  name: "inception_5b/double3x3_reduce/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
 bottom: "inception_5b/double3x3_reduce/bn"
  top: "inception_5b/double3x3a"
  name: "inception_5b/double3x3a"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 224
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_5b/double3x3a"
  name: "inception_5b/double3x3a/bn"
  top: "inception_5b/double3x3a/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_5b/double3x3a/bn"
  top: "inception_5b/double3x3a/bn"
  name: "inception_5b/double3x3a/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
 bottom: "inception_5b/double3x3a/bn"
  top: "inception_5b/double3x3b"
  name: "inception_5b/double3x3b"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 224
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_5b/double3x3b"
  name: "inception_5b/double3x3b/bn"
  top: "inception_5b/double3x3b/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_5b/double3x3b/bn"
  top: "inception_5b/double3x3b/bn"
  name: "inception_5b/double3x3b/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_5a/output"
  top: "inception_5b/pool"
  name: "inception_5b/pool"
  type: "Pooling"
  pooling_param {
    engine: MKL2017
    pool: MAX
    kernel_size: 3
    stride: 1
    pad: 1
  }
}
layer {
 bottom: "inception_5b/pool"
  top: "inception_5b/pool_proj"
  name: "inception_5b/pool_proj"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    engine: MKL2017
    num_output: 128
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  bottom: "inception_5b/pool_proj"
  name: "inception_5b/pool_proj/bn"
  top: "inception_5b/pool_proj/bn"
  type: "BatchNorm"
  batch_norm_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_5b/pool_proj/bn"
  top: "inception_5b/pool_proj/bn"
  name: "inception_5b/pool_proj/bn/relu"
  type: "ReLU"
  relu_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_5b/1x1/bn"
  bottom: "inception_5b/3x3/bn"
  bottom: "inception_5b/double3x3b/bn"
  bottom: "inception_5b/pool_proj/bn"
  top: "inception_5b/output"
  name: "inception_5b/output"
  type: "Concat"
  concat_param {
    engine: MKL2017
  }
}
layer {
  bottom: "inception_5b/output"
  top: "pool5/7x7_s1"
  name: "pool5/7x7_s1"
  type: "Pooling"
  pooling_param {
    engine: MKL2017
    pool: AVE
    kernel_size: 7
    stride: 1
  }
}
layer {
  bottom: "pool5/7x7_s1"
  top: "loss3/classifier"
  name: "loss3/classifier"
  type: "InnerProduct"
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
  bottom: "loss3/classifier"
  bottom: "label"
  top: "loss3/loss"
  name: "loss3/loss"
  type: "SoftmaxWithLoss"
  loss_weight: 1
}
layer {
  bottom: "loss3/classifier"
  top: "loss3/prob"
  name: "loss3/prob"
  type: "Softmax"
  include {
    phase: TEST
  }
}
layer {
  bottom: "loss3/prob"
  bottom: "label"
  top: "loss3/top-1"
  name: "loss3/top-1"
  type: "Accuracy"
  include {
    phase: TEST
  }
}
layer {
  bottom: "loss3/prob"
  bottom: "label"
  top: "loss3/top-5"
  name: "loss3/top-5"
  type: "Accuracy"
  accuracy_param {
    top_k: 5
  }
  include {
    phase: TEST
  }
}
     """.stripMargin
}

object GoogleNet_v2 {
  def apply(classNum: Int): Module[Float] = {
    val features1 = new Sequential()
    features1.add(new SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3, propagateBack = true)
      .setName("conv1/7x7_s2")
        .setInitMethod(Xavier))
    features1.add(new BatchNormalization(64, 1e-5).setName("conv1/7x7_s2/bn"))
    features1.add(new ReLU(ip = false).setName("conv1/7x7_s2/bn/sc/relu"))
    features1.add(new SpatialMaxPooling(3, 3, 2, 2).ceil().setName("pool1/3x3_s2"))
    features1.add(new SpatialConvolution(64, 64, 1, 1).setName("conv2/3x3_reduce")
      .setInitMethod(Xavier))
    features1.add(new BatchNormalization(64, 1e-5).setName("conv2/3x3_reduce/bn"))
    features1.add(new ReLU(ip = false).setName("conv2/3x3_reduce/bn/sc/relu"))
    features1.add(new SpatialConvolution(64, 192, 3, 3, 1, 1, 1, 1).setName("conv2/3x3")
        .setInitMethod(Xavier))
    features1.add(new BatchNormalization(192, 1e-5).setName("conv2/3x3/bn"))
    features1.add(new ReLU(ip = false).setName("conv2/3x3/bn/sc/relu"))
    features1.add(new SpatialMaxPooling(3, 3, 2, 2).ceil().setName("pool2/3x3_s2"))
    features1.add(inception(192, T(T(64), T(64, 64), T(64, 96), T("avg", 32)), "inception_3a/"))
    features1.add(inception(256, T(T(64), T(64, 96), T(64, 96), T("avg", 64)), "inception_3b/"))
    features1.add(inception(320, T(T(0), T(128, 160), T(64, 96), T("max", 0)), "inception_3c/"))

    val output1 = new Sequential()
    output1.add(new SpatialAveragePooling(5, 5, 3, 3).ceil().setName("pool3/5x5_s3"))
    output1.add(new SpatialConvolution(576, 128, 1, 1, 1, 1).setName("loss1/conv")
        .setInitMethod(Xavier))
    output1.add(new BatchNormalization(128, 1e-5).setName("loss1/conv/bn"))
    output1.add(new ReLU(ip = false).setName("loss1/conv/bn/relu"))
    output1.add(new View(128 * 4 * 4).setNumInputDims(3))
    output1.add(new Linear(128 * 4 * 4, 1024).setName("loss1/fc").setInitMethod(Xavier))
    output1.add(new ReLU(ip = false).setName("loss1/fc/bn/relu"))
    output1.add(new Linear(1024, classNum).setName("loss1/classifier").setInitMethod(Xavier))
    //    output1.add(LogSoftMax.setName("loss1/loss"))

    val features2 = new Sequential()
    features2.add(inception(576, T(T(224), T(64, 96), T(96, 128), T("avg", 128)), "inception_4a/"))
    features2.add(inception(576, T(T(192), T(96, 128), T(96, 128), T("avg", 128)), "inception_4b/"))
    features2.add(inception(576, T(T(160), T(128, 160), T(128, 160), T("avg", 96)),
      "inception_4c/"))
    features2.add(inception(576, T(T(96), T(128, 192), T(160, 192), T("avg", 96)), "inception_4d/"))
    features2.add(inception(576, T(T(0), T(128, 192), T(192, 256), T("max", 0)), "inception_4e/"))

    val output2 = new Sequential()
    output2.add(new SpatialAveragePooling(5, 5, 3, 3).ceil().setName("pool4/5x5_s3"))
    output2.add(new SpatialConvolution(1024, 128, 1, 1, 1, 1).setName("loss2/conv")
        .setInitMethod(Xavier))
    output2.add(new BatchNormalization(128, 1e-5).setName("loss2/conv/bn"))
    output2.add(new ReLU(ip = false).setName("loss2/conv/bn/relu"))
    output2.add(new View(128 * 2 * 2).setNumInputDims(3))
    output2.add(new Linear(128 * 2 * 2, 1024).setName("loss2/fc").setInitMethod(Xavier))
    output2.add(new ReLU(ip = false).setName("loss2/fc/bn/relu"))
    output2.add(new Linear(1024, classNum).setName("loss2/classifier").setInitMethod(Xavier))
    //    output2.add(LogSoftMax.setName("loss2/loss"))

    val output3 = new Sequential()
    output3.add(inception(1024, T(T(352), T(192, 320), T(160, 224), T("avg", 128)),
      "inception_5a/"))
    output3.add(inception(1024, T(T(352), T(192, 320), T(192, 224), T("max", 128)),
      "inception_5b/"))
    output3.add(new SpatialAveragePooling(7, 7, 1, 1).ceil().setName("pool5/7x7_s1"))
    output3.add(new View(1024).setNumInputDims(3))
    output3.add(new Linear(1024, classNum).setName("loss3/classifier").setInitMethod(Xavier))
    //    output3.add(LogSoftMax.setName("loss3/loss"))

    val split2 = new Concat(2)
    split2.add(output3)
    split2.add(output2)

    val mainBranch = new Sequential()
    mainBranch.add(features2)
    mainBranch.add(split2)

    val split1 = new Concat(2)
    split1.add(mainBranch)
    split1.add(output1)

    val model = new Sequential()

    model.add(features1)
    model.add(split1)

    model.reset()
    model
  }

  def inception(inputSize: Int, config: Table, namePrefix: String): Module[Float] = {
    val concat = new Concat(2)
    if (config[Table](1)[Int](1) != 0) {
      val conv1 = new Sequential()
      conv1.add(new SpatialConvolution(inputSize, config[Table](1)(1), 1, 1, 1, 1)
          .setName(namePrefix + "1x1")
          .setInitMethod(Xavier))
      conv1.add(new BatchNormalization(config[Table](1)(1), 1e-5) .setName(namePrefix + "1x1/bn"))
      conv1.add(new ReLU(ip = false).setName(namePrefix + "1x1/bn/sc/relu"))
      concat.add(conv1)
    }

    val conv3 = new Sequential()
    conv3.add(new SpatialConvolution(inputSize, config[Table](2)(1), 1, 1, 1, 1)
        .setName(namePrefix + "3x3_reduce")
        .setInitMethod(Xavier))
    conv3.add(new BatchNormalization(config[Table](2)(1), 1e-5)
      .setName(namePrefix + "3x3_reduce/bn"))
    conv3.add(new ReLU(ip = false).setName(namePrefix + "3x3_reduce/bn/sc/relu"))
    if (config[Table](4)[String](1) == "max" && config[Table](4)[Int](2) == 0) {
      conv3.add(new SpatialConvolution(config[Table](2)(1), config[Table](2)(2), 3, 3, 2, 2, 1, 1)
          .setName(namePrefix + "3x3")
          .setInitMethod(Xavier))
    } else {
      conv3.add(new SpatialConvolution(config[Table](2)(1), config[Table](2)(2), 3, 3, 1, 1, 1, 1)
          .setName(namePrefix + "3x3")
          .setInitMethod(Xavier))
    }
    conv3.add(new BatchNormalization(config[Table](2)(2), 1e-5).setName(namePrefix + "3x3/bn"))
    conv3.add(new ReLU(ip = false).setName(namePrefix + "3x3/bn/sc/relu"))
    concat.add(conv3)

    val conv3xx = new Sequential()
    conv3xx.add(new SpatialConvolution(inputSize, config[Table](3)(1), 1, 1, 1, 1)
        .setName(namePrefix + "double3x3_reduce")
        .setInitMethod(Xavier))
    conv3xx.add(new BatchNormalization(config[Table](3)(1), 1e-5)
      .setName(namePrefix + "double3x3_reduce/bn"))
    conv3xx.add(new ReLU(ip = false).setName(namePrefix + "double3x3_reduce/bn/sc/relu"))

    conv3xx.add(new SpatialConvolution(config[Table](3)(1), config[Table](3)(2), 3, 3, 1, 1, 1, 1)
        .setName(namePrefix + "double3x3a")
        .setInitMethod(Xavier))
    conv3xx.add(new BatchNormalization(config[Table](3)(2), 1e-5)
      .setName(namePrefix + "double3x3a/bn"))
    conv3xx.add(new ReLU(ip = false).setName(namePrefix + "double3x3a/bn/sc/relu"))

    if (config[Table](4)[String](1) == "max" && config[Table](4)[Int](2) == 0) {
      conv3xx.add(new SpatialConvolution(config[Table](3)(2), config[Table](3)(2), 3, 3, 2, 2, 1, 1)
          .setName(namePrefix + "double3x3b")
          .setInitMethod(Xavier))
    } else {
      conv3xx.add(new SpatialConvolution(config[Table](3)(2), config[Table](3)(2), 3, 3, 1, 1, 1, 1)
          .setName(namePrefix + "double3x3b")
          .setInitMethod(Xavier))
    }
    conv3xx.add(new BatchNormalization(config[Table](3)(2), 1e-5)
      .setName(namePrefix + "double3x3b/bn"))
    conv3xx.add(new ReLU(ip = false).setName(namePrefix + "double3x3b/bn/sc/relu"))
    concat.add(conv3xx)

    val pool = new Sequential()
    config[Table](4)[String](1) match {
      case "max" =>
        if (config[Table](4)[Int](2) != 0) {
          pool.add(new SpatialMaxPooling(3, 3, 1, 1, 1, 1).ceil().setName(namePrefix + "pool"))
        } else {
          pool.add(new SpatialMaxPooling(3, 3, 2, 2).ceil().setName(namePrefix + "pool"))
        }
      case "avg" =>
        pool.add(new
          SpatialAveragePooling(3, 3, 1, 1, 1, 1).ceil().setName(namePrefix + "pool"))
      case _ => throw new IllegalArgumentException
    }

    if (config[Table](4)[Int](2) != 0) {
      pool.add(new SpatialConvolution(inputSize, config[Table](4)[Int](2), 1, 1, 1, 1)
          .setName(namePrefix + "pool_proj")
          .setInitMethod(Xavier))
      pool.add(new BatchNormalization(config[Table](4)(2), 1e-5)
        .setName(namePrefix + "pool_proj/bn"))
      pool.add(new ReLU(ip = false).setName(namePrefix + "pool_proj/bn/sc/relu"))
    }
    concat.add(pool)
    concat.setName(namePrefix + "output")
    concat
  }
}

