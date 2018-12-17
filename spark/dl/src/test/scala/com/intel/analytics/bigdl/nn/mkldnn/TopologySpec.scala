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

package com.intel.analytics.bigdl.nn.mkldnn

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.nn.mkldnn.Phase.TrainingPhase
import com.intel.analytics.bigdl.numeric.NumericFloat
import org.scalatest.{FlatSpec, Matchers}

class TopologySpec extends FlatSpec with Matchers {

  "LeNet5 has no tanh" should "work correctly" in {
    val inputShape = Array(4, 1, 28, 28)
    val outputShape = Array(4, 10)
    val prototxt = s"""
         |name: "LeNet"
         |force_backward: true
         |layer {
         |  name: "data"
         |  type: "DummyData"
         |  top: "data"
         |  dummy_data_param {
         |    data_filler {
         |      type: "xavier"
         |    }
         |    shape: { ${shape2Dim(inputShape)} }
         |  }
         |}
         |layer {
         |  name: "conv1"
         |  type: "Convolution"
         |  bottom: "data"
         |  top: "conv1"
         |  param {
         |    lr_mult: 1
         |  }
         |  param {
         |    lr_mult: 2
         |  }
         |  convolution_param {
         |    num_output: 20
         |    kernel_size: 5
         |    stride: 1
         |    weight_filler {
         |      type: "msra"
         |      variance_norm: FAN_OUT
         |    }
         |    bias_filler {
         |      type: "constant"
         |      value: 0
         |    }
         |  }
         |}
         |layer {
         |  name: "pool1"
         |  type: "Pooling"
         |  bottom: "conv1"
         |  top: "pool1"
         |  pooling_param {
         |    pool: MAX
         |    kernel_size: 2
         |    stride: 2
         |  }
         |}
         |layer {
         |  name: "conv2"
         |  type: "Convolution"
         |  bottom: "pool1"
         |  top: "conv2"
         |  param {
         |    lr_mult: 1
         |  }
         |  param {
         |    lr_mult: 2
         |  }
         |  convolution_param {
         |    num_output: 50
         |    kernel_size: 5
         |    stride: 1
         |    weight_filler {
         |      type: "xavier"
         |    }
         |    bias_filler {
         |      type: "constant"
         |    }
         |  }
         |}
         |layer {
         |  name: "pool2"
         |  type: "Pooling"
         |  bottom: "conv2"
         |  top: "pool2"
         |  pooling_param {
         |    pool: MAX
         |    kernel_size: 2
         |    stride: 2
         |  }
         |}
         |layer {
         |  name: "ip1"
         |  type: "InnerProduct"
         |  bottom: "pool2"
         |  top: "ip1"
         |  param {
         |    lr_mult: 1
         |  }
         |  param {
         |    lr_mult: 2
         |  }
         |  inner_product_param {
         |    num_output: 500
         |    weight_filler {
         |      type: "xavier"
         |    }
         |    bias_filler {
         |      type: "constant"
         |    }
         |  }
         |}
         |layer {
         |  name: "relu1"
         |  type: "ReLU"
         |  bottom: "ip1"
         |  top: "ip1"
         |}
         |layer {
         |  name: "ip2"
         |  type: "InnerProduct"
         |  bottom: "ip1"
         |  top: "ip2"
         |  param {
         |    lr_mult: 1
         |  }
         |  param {
         |    lr_mult: 2
         |  }
         |  inner_product_param {
         |    num_output: 10
         |    weight_filler {
         |      type: "xavier"
         |    }
         |    bias_filler {
         |      type: "constant"
         |    }
         |  }
         |}
       """.stripMargin
//    |layer {
//      |  name: "prob"
//      |  type: "Softmax"
//      |  bottom: "ip2"
//      |  top: "prob"
//      |}
//    |

    val bigdl = Sequential()
      .add(SpatialConvolution(1, 20, 5, 5).setName("conv1"))
      .add(MaxPooling(2, 2, 2, 2).setName("pool1"))
      .add(SpatialConvolution(20, 50, 5, 5).setName("conv2"))
      .add(MaxPooling(2, 2, 2, 2).setName("pool2"))
      .add(Linear(50 * 4 * 4, 500).setName("ip1"))
      .add(ReLU().setName("relu1"))
      .add(Linear(500, 10).setName("ip2"))
      .add(ReorderMemory(HeapData(outputShape, Memory.Format.nc)))
//      .add(SoftMax().setName("prob")) // TODO SoftMax is totally different with Caffe.
    bigdl.compile(TrainingPhase, Array(HeapData(inputShape, Memory.Format.nchw)))

    Tools.compare(prototxt, bigdl, inputShape, outputShape, 1e-6)
  }

  "eltwise" should "work correctly" in {
    val nInput = 3
    val nOutput = 2
    val inputShape = Array(4, 3, 5, 5)
    val outputShape = Array(4, 2, 3, 3)

    val kernel = 3
    val pad = 1
    val stride = 2

    val prototxt =
      s"""
          | name: "eltwise-simple"
          |force_backward: true
          |layer {
          |  name: "data"
          |  type: "DummyData"
          |  top: "data"
          |  include {
          |    phase: TRAIN
          |  }
          |  dummy_data_param {
          |    data_filler {
          |      type: "xavier"
          |    }
          |    shape: { dim: 4 dim: 3 dim: 5 dim: 5 }
          |  }
          |}
          |layer {
          |  bottom: "data"
          |  top: "conv1"
          |  name: "conv1"
          |  type: "Convolution"
          |  convolution_param {
          |    num_output: 2
          |    kernel_size: 3
          |    pad: 1
          |    stride: 2
          |    weight_filler {
          |      # type: "msra"
          |      # variance_norm: FAN_OUT
          |      type: "constant"
          |      value: 0.1
          |    }
          |    bias_filler {
          |      # type: "gaussian"
          |      type: "constant"
          |      value: 0.1
          |    }
          |  }
          |}
          |layer {
          |  bottom: "data"
          |  top: "conv2"
          |  name: "conv2"
          |  type: "Convolution"
          |  convolution_param {
          |    num_output: 2
          |    kernel_size: 3
          |    pad: 1
          |    stride: 2
          |    weight_filler {
          |      # type: "msra"
          |      # variance_norm: FAN_OUT
          |      type: "constant"
          |      value: 0.1
          |    }
          |    bias_filler {
          |      # type: "gaussian"
          |      type: "constant"
          |      value: 0.1
          |    }
          |  }
          |}
          |layer {
          |  bottom: "conv1"
          |  bottom: "conv2"
          |  top: "eltwise"
          |  name: "eltwise"
          |  type: "Eltwise"
          |  eltwise_param {
          |  }
          |}
          |
       """.stripMargin

    val conv1 = SpatialConvolution(nInput, nOutput, kernel, kernel, stride, stride, pad, pad, 1)
      .setName("conv1")
    val conv2 = SpatialConvolution(nInput, nOutput, kernel, kernel, stride, stride, pad, pad, 1)
      .setName("conv2")
    val model = Sequential()
      .add(ConcatTable().add(conv2).add(conv1))
      .add(CAddTable().setName("eltwise"))
      .add(ReorderMemory(HeapData(outputShape, Memory.Format.nchw)))

    model.compile(TrainingPhase, Array(HeapData(inputShape, Memory.Format.nchw)))

    Tools.compare(prototxt, model, inputShape, outputShape)
  }

  "resnet 50" should "work correctly" in {
    val prototxt =
      s"""
         |name: "ResNet-50"
         |force_backward: true
         |layer {
         |  name: "data"
         |  type: "DummyData"
         |  top: "data"
         |  top: "label"
         |  include {
         |    phase: TRAIN
         |  }
         |  dummy_data_param {
         |    data_filler {
         |      type: "constant"
         |      value: 0.01
         |    }
         |    shape: { dim: 4 dim: 3 dim: 224 dim: 224 }
         |    shape: { dim: 4 dim: 1 dim: 1 dim: 1  }
         |  }
         |}
         |
         |layer {
         |  name: "data"
         |  type: "DummyData"
         |  top: "data"
         |  top: "label"
         |  include {
         |    phase: TEST
         |  }
         |  dummy_data_param {
         |    data_filler {
         |      type: "constant"
         |      value: 0.01
         |    }
         |    shape: { dim: 32 dim: 3 dim: 224 dim: 224 }
         |    shape: { dim: 32 dim: 1 dim: 1   dim: 1   }
         |  }
         |}
         |
         |layer {
         |  bottom: "data"
         |  top: "conv1"
         |  name: "conv1"
         |  type: "Convolution"
         |  convolution_param {
         |    num_output: 64
         |    kernel_size: 7
         |    pad: 3
         |    stride: 2
         |    weight_filler {
         |      type: "msra"
         |      variance_norm: FAN_OUT
         |    }
         |    bias_filler {
         |      type: "constant"
         |      value: 0
         |    }
         |  }
         |}
         |
         |# layer {
         |#   bottom: "conv1"
         |#   top: "conv1"
         |#   name: "bn_conv1"
         |#   type: "BatchNorm"
         |#   param { lr_mult: 0 }
         |#   param { lr_mult: 0 }
         |#   param { lr_mult: 0 }
         |#   batch_norm_param {
         |#     moving_average_fraction: 0.9
         |#     filler { value: 1 }
         |#   }
         |# }
         |
         |layer {
         |  bottom: "conv1"
         |  top: "conv1"
         |  name: "scale_conv1"
         |  type: "Scale"
         |  param { decay_mult: 0 }
         |  param { decay_mult: 0 }
         |  scale_param {
         |    bias_term: true
         |  }
         |}
         |
         |layer {
         |  bottom: "conv1"
         |  top: "conv1"
         |  name: "conv1_relu"
         |  type: "ReLU"
         |  relu_param {
         |  }
         |}
         |
         |layer {
         |  bottom: "conv1"
         |  top: "pool1"
         |  name: "pool1"
         |  type: "Pooling"
         |  pooling_param {
         |    kernel_size: 3
         |    stride: 2
         |    pool: MAX
         |  }
         |}
       """.stripMargin
    val inputShape = Array(4, 3, 224, 224)
    val outputShape = Array(4, 64, 56, 56)

    val model = Sequential()
      .add(SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3, propagateBack = true).setName("conv1"))
      .add(ReLU().setName("conv1_relu"))
      .add(MaxPooling(3, 3, 2, 2).setName("pool1"))
      .add(ReorderMemory(HeapData(outputShape, Memory.Format.nchw)))
    model.compile(TrainingPhase, Array(HeapData(inputShape, Memory.Format.nchw)))

    Tools.compare(prototxt, model, inputShape, outputShape)
  }

  "bottleneck" should "work correctly" in {
    val prototxt =
      s"""
         |name: "ResNet-50"
         |force_backward: true
         |layer {
         |  name: "data"
         |  type: "DummyData"
         |  top: "data"
         |  top: "label"
         |  include {
         |    phase: TRAIN
         |  }
         |  dummy_data_param {
         |    data_filler {
         |      type: "constant"
         |      value: 0.01
         |    }
         |    shape: { dim: 4 dim: 3 dim: 224 dim: 224 }
         |    shape: { dim: 4 dim: 1 dim: 1 dim: 1  }
         |  }
         |}
         |
         |layer {
         |  bottom: "data"
         |  top: "conv1"
         |  name: "conv1"
         |  type: "Convolution"
         |  convolution_param {
         |    num_output: 64
         |    kernel_size: 7
         |    pad: 3
         |    stride: 2
         |    weight_filler {
         |      type: "msra"
         |      variance_norm: FAN_OUT
         |    }
         |    bias_filler {
         |      type: "constant"
         |      value: 0
         |    }
         |  }
         |}
         |
         |layer {
         |  bottom: "conv1"
         |  top: "conv1_relu" # delete inplace
         |  name: "conv1_relu"
         |  type: "ReLU"
         |  relu_param {
         |    fuse: false
         |  }
         |}
         |
         |layer {
         |  bottom: "conv1_relu"
         |  top: "pool1"
         |  name: "pool1"
         |  type: "Pooling"
         |  pooling_param {
         |    kernel_size: 3
         |    stride: 2
         |    pool: MAX
         |  }
         |}
         |
         |layer {
         |  bottom: "pool1"
         |  top: "res2a_branch1"
         |  name: "res2a_branch1"
         |  type: "Convolution"
         |  convolution_param {
         |    num_output: 256
         |    kernel_size: 1
         |    pad: 0
         |    stride: 1
         |    bias_term: true # change to true
         |    weight_filler {
         |      type: "msra"
         |    }
         |    bias_filler {
         |      type: "constant"
         |      value: 0
         |    }
         |  }
         |}
         |
         |layer {
         |  bottom: "res2a_branch1"
         |  top: "res2a_branch1"
         |  name: "scale2a_branch1"
         |  type: "Scale"
         |  param { decay_mult: 0 }
         |  param { decay_mult: 0 }
         |  scale_param {
         |    bias_term: true
         |  }
         |}
         |
         |layer {
         |  bottom: "pool1"
         |  top: "res2a_branch2a"
         |  name: "res2a_branch2a"
         |  type: "Convolution"
         |  convolution_param {
         |
         |    num_output: 64
         |    kernel_size: 1
         |    pad: 0
         |    stride: 1
         |    bias_term: true # change to true.
         |    weight_filler {
         |      type: "msra"
         |    }
         |    bias_filler {
         |      type: "constant"
         |      value: 0
         |    }
         |  }
         |}
         |
         |layer {
         |  bottom: "res2a_branch2a"
         |  top: "res2a_branch2a"
         |  name: "scale2a_branch2a"
         |  type: "Scale"
         |  param { decay_mult: 0 }
         |  param { decay_mult: 0 }
         |  scale_param {
         |    bias_term: true
         |  }
         |}
         |
         |layer {
         |  bottom: "res2a_branch2a"
         |  top: "res2a_branch2a"
         |  name: "res2a_branch2a_relu"
         |  type: "ReLU"
         |  relu_param {
         |
         |  }
         |}
         |
         |layer {
         |  bottom: "res2a_branch2a"
         |  top: "res2a_branch2b"
         |  name: "res2a_branch2b"
         |  type: "Convolution"
         |  convolution_param {
         |    num_output: 64
         |    kernel_size: 3
         |    pad: 1
         |    stride: 1
         |    bias_term: true # change to true
         |    weight_filler {
         |      type: "msra"
         |    }
         |    bias_filler {
         |      type: "constant"
         |      value: 0
         |    }
         |  }
         |}
         |
         |layer {
         |  bottom: "res2a_branch2b"
         |  top: "res2a_branch2b"
         |  name: "scale2a_branch2b"
         |  type: "Scale"
         |  param { decay_mult: 0 }
         |  param { decay_mult: 0 }
         |  scale_param {
         |    bias_term: true
         |  }
         |}
         |
         |layer {
         |  bottom: "res2a_branch2b"
         |  top: "res2a_branch2b"
         |  name: "res2a_branch2b_relu"
         |  type: "ReLU"
         |  relu_param {
         |
         |  }
         |}
         |
         |layer {
         |  bottom: "res2a_branch2b"
         |  top: "res2a_branch2c"
         |  name: "res2a_branch2c"
         |  type: "Convolution"
         |  convolution_param {
         |    num_output: 256
         |    kernel_size: 1
         |    pad: 0
         |    stride: 1
         |    bias_term: true # change to true
         |    weight_filler {
         |      type: "msra"
         |    }
         |    bias_filler {
         |      type: "constant"
         |      value: 0
         |    }
         |  }
         |}
         |
         |layer {
         |  bottom: "res2a_branch2c"
         |  top: "res2a_branch2c"
         |  name: "scale2a_branch2c"
         |  type: "Scale"
         |  param { decay_mult: 0 }
         |  param { decay_mult: 0 }
         |  scale_param {
         |    bias_term: true
         |  }
         |}
         |
         |layer {
         |  bottom: "res2a_branch1"
         |  bottom: "res2a_branch2c"
         |  top: "res2a"
         |  name: "res2a"
         |  type: "Eltwise"
         |  eltwise_param {
         |
         |  }
         |}
         |
         |layer {
         |  bottom: "res2a"
         |  top: "res2a"
         |  name: "res2a_relu"
         |  type: "ReLU"
         |  relu_param {
         |
         |  }
         |}
         |layer {
         |  bottom: "res2a"
         |  top: "res2b_branch2a"
         |  name: "res2b_branch2a"
         |  type: "Convolution"
         |  convolution_param {
         |    num_output: 64
         |    kernel_size: 1
         |    pad: 0
         |    stride: 1
         |    bias_term: true # change to true
         |    weight_filler {
         |      type: "msra"
         |    }
         |    bias_filler {
         |      type: "constant"
         |      value: 0
         |    }
         |  }
         |}
         |
         |layer {
         |  bottom: "res2b_branch2a"
         |  top: "res2b_branch2a"
         |  name: "scale2b_branch2a"
         |  type: "Scale"
         |  param { decay_mult: 0 }
         |  param { decay_mult: 0 }
         |  scale_param {
         |    bias_term: true
         |  }
         |}
         |
         |layer {
         |  bottom: "res2b_branch2a"
         |  top: "res2b_branch2a"
         |  name: "res2b_branch2a_relu"
         |  type: "ReLU"
         |  relu_param {
         |
         |  }
         |}
         |
         |layer {
         |  bottom: "res2b_branch2a"
         |  top: "res2b_branch2b"
         |  name: "res2b_branch2b"
         |  type: "Convolution"
         |  convolution_param {
         |    num_output: 64
         |    kernel_size: 3
         |    pad: 1
         |    stride: 1
         |    bias_term: true # change to true
         |    weight_filler {
         |      type: "msra"
         |    }
         |    bias_filler {
         |      type: "constant"
         |      value: 0
         |    }
         |  }
         |}
         |
         |layer {
         |  bottom: "res2b_branch2b"
         |  top: "res2b_branch2b"
         |  name: "scale2b_branch2b"
         |  type: "Scale"
         |  param { decay_mult: 0 }
         |  param { decay_mult: 0 }
         |  scale_param {
         |    bias_term: true
         |  }
         |}
         |
         |layer {
         |  bottom: "res2b_branch2b"
         |  top: "res2b_branch2b"
         |  name: "res2b_branch2b_relu"
         |  type: "ReLU"
         |  relu_param {
         |
         |  }
         |}
         |
         |layer {
         |  bottom: "res2b_branch2b"
         |  top: "res2b_branch2c"
         |  name: "res2b_branch2c"
         |  type: "Convolution"
         |  convolution_param {
         |    num_output: 256
         |    kernel_size: 1
         |    pad: 0
         |    stride: 1
         |    bias_term: true # change to true
         |    weight_filler {
         |      type: "msra"
         |    }
         |    bias_filler {
         |      type: "constant"
         |      value: 0
         |    }
         |  }
         |}
         |
         |layer {
         |  bottom: "res2b_branch2c"
         |  top: "res2b_branch2c"
         |  name: "scale2b_branch2c"
         |  type: "Scale"
         |  param { decay_mult: 0 }
         |  param { decay_mult: 0 }
         |  scale_param {
         |    bias_term: true
         |  }
         |}
         |
         |layer {
         |  bottom: "res2a"
         |  bottom: "res2b_branch2c"
         |  top: "res2b"
         |  name: "res2b"
         |  type: "Eltwise"
         |  eltwise_param {
         |
         |  }
         |}
         |
         |layer {
         |  bottom: "res2b"
         |  top: "res2b"
         |  name: "res2b_relu"
         |  type: "ReLU"
         |  relu_param {
         |
         |  }
         |}
         |
         |layer {
         |  bottom: "res2b"
         |  top: "res2c_branch2a"
         |  name: "res2c_branch2a"
         |  type: "Convolution"
         |  convolution_param {
         |
         |    num_output: 64
         |    kernel_size: 1
         |    pad: 0
         |    stride: 1
         |    bias_term: true # change to true
         |    weight_filler {
         |      type: "msra"
         |    }
         |    bias_filler {
         |      type: "constant"
         |      value: 0
         |    }
         |  }
         |}
         |
         |layer {
         |  bottom: "res2c_branch2a"
         |  top: "res2c_branch2a"
         |  name: "scale2c_branch2a"
         |  type: "Scale"
         |  param { decay_mult: 0 }
         |  param { decay_mult: 0 }
         |  scale_param {
         |    bias_term: true
         |  }
         |}
         |
         |layer {
         |  bottom: "res2c_branch2a"
         |  top: "res2c_branch2a"
         |  name: "res2c_branch2a_relu"
         |  type: "ReLU"
         |  relu_param {
         |
         |  }
         |}
         |
         |layer {
         |  bottom: "res2c_branch2a"
         |  top: "res2c_branch2b"
         |  name: "res2c_branch2b"
         |  type: "Convolution"
         |  convolution_param {
         |    num_output: 64
         |    kernel_size: 3
         |    pad: 1
         |    stride: 1
         |    bias_term: true # change to true
         |    weight_filler {
         |      type: "msra"
         |    }
         |    bias_filler {
         |      type: "constant"
         |      value: 0
         |    }
         |  }
         |}
         |
         |layer {
         |  bottom: "res2c_branch2b"
         |  top: "res2c_branch2b"
         |  name: "scale2c_branch2b"
         |  type: "Scale"
         |  param { decay_mult: 0 }
         |  param { decay_mult: 0 }
         |  scale_param {
         |    bias_term: true
         |  }
         |}
         |
         |layer {
         |  bottom: "res2c_branch2b"
         |  top: "res2c_branch2b"
         |  name: "res2c_branch2b_relu"
         |  type: "ReLU"
         |  relu_param {
         |
         |  }
         |}
         |
         |layer {
         |  bottom: "res2c_branch2b"
         |  top: "res2c_branch2c"
         |  name: "res2c_branch2c"
         |  type: "Convolution"
         |  convolution_param {
         |
         |    num_output: 256
         |    kernel_size: 1
         |    pad: 0
         |    stride: 1
         |    bias_term: true # change to true
         |    weight_filler {
         |      type: "msra"
         |    }
         |    bias_filler {
         |      type: "constant"
         |      value: 0
         |    }
         |  }
         |}
         |
         |layer {
         |  bottom: "res2c_branch2c"
         |  top: "res2c_branch2c"
         |  name: "scale2c_branch2c"
         |  type: "Scale"
         |  param { decay_mult: 0 }
         |  param { decay_mult: 0 }
         |  scale_param {
         |    bias_term: true
         |  }
         |}
         |
         |layer {
         |  bottom: "res2b"
         |  bottom: "res2c_branch2c"
         |  top: "res2c"
         |  name: "res2c"
         |  type: "Eltwise"
         |  eltwise_param {
         |
         |  }
         |}
         |
         |layer {
         |  bottom: "res2c"
         |  top: "res2c_" # do not do inplace
         |  name: "res2c_relu"
         |  type: "ReLU"
         |  relu_param {
         |
         |  }
         |}
       """.stripMargin
    val inputShape = Array(4, 3, 224, 224)
    val outputShape = Array(4, 256, 56, 56)

    val model = ResNet50.getModel(inputShape, outputShape)
    model.compile(TrainingPhase, Array(HeapData(inputShape, Memory.Format.nchw)))

    Tools.compare(prototxt, model, inputShape, outputShape, 1e-5)
  }

  object ResNet50 {
    var iChannels = 64

    def shortcut(nInputPlane: Int, nOutputPlane: Int, stride: Int, name: String): Module[Float] = {
      val useConv = nInputPlane != nOutputPlane

      if (useConv) {
        Sequential()
          .add(SpatialConvolution(nInputPlane, nOutputPlane, 1, 1, stride, stride)
            .setName(s"res${name}_branch1"))
      } else if (nInputPlane != nOutputPlane) {
        throw new IllegalArgumentException(s"useConv false")
      } else {
        Identity()
      }
    }

    def bottleneck(n: Int, stride: Int, name: String = ""): Module[Float] = {
      val nInputPlane = iChannels
      iChannels = n * 4

      val s = Sequential()
      s.add(SpatialConvolution(nInputPlane, n, 1, 1, 1, 1, 0, 0).setName(s"res${name}_branch2a"))
        .add(ReLU().setName(s"res${name}_branch2a_relu"))
        .add(SpatialConvolution(n, n, 3, 3, stride, stride, 1, 1).setName(s"res${name}_branch2b"))
        .add(ReLU().setName(s"res${name}_branch2b_relu"))
        .add(SpatialConvolution(n, n*4, 1, 1, 1, 1, 0, 0).setName(s"res${name}_branch2c"))

      val model = Sequential()
        .add(ConcatTable().
          add(s).
          add(shortcut(nInputPlane, n*4, stride, name)).setName(s"$name/concatTable"))
        .add(CAddTable().setName(s"res$name"))
        .add(ReLU().setName(s"res${name}_relu"))
      model
    }

    def layer(block: (Int, Int, String) => Module[Float], features: Int,
      count: Int, stride: Int = 1, name : String): Module[Float] = {
      val s = Sequential()
      for (i <- 1 to count) {
        s.add(block(features, if (i == 1) stride else 1, getName(i, name)))
      }
      s
    }

    def getName(i: Int, name: String): String = {
      i match {
        case 1 => name + "a"
        case 2 => name + "b"
        case 3 => name + "c"
        case 4 => name + "d"
        case 5 => name + "e"
        case 6 => name + "f"
      }
    }

    def getModel(inputShape: Array[Int], outputShape: Array[Int]): MklDnnContainer = {
      iChannels = 64

      Sequential()
        .add(SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3).setName("conv1").setReLU(true))
        .add(ReLU().setName("conv1_relu"))
        .add(MaxPooling(3, 3, 2, 2).setName("pool1"))
        .add(layer(bottleneck, 64, 3, name = "2"))
        .add(ReorderMemory(HeapData(outputShape, Memory.Format.nchw)))
    }
  }

  private def shape2Dim(shape: Array[Int]): String = {
    shape.map(x => "dim: " + x).mkString(" ")
  }
}
