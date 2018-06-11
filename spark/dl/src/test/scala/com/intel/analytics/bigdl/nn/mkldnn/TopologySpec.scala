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

import com.intel.analytics.bigdl.nn
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

    val bigdl = nn.Sequential()
      .add(ConvolutionDnn(1, 20, 5, 5).setName("conv1"))
      .add(PoolingDnn(2, 2, 2, 2).setName("pool1"))
      .add(ConvolutionDnn(20, 50, 5, 5).setName("conv2"))
      .add(PoolingDnn(2, 2, 2, 2).setName("pool2"))
      .add(Linear(50 * 4 * 4, 500).setName("ip1"))
      .add(ReLUDnn(false).setName("relu1"))
      .add(Linear(500, 10).setName("ip2"))
//      .add(nn.SoftMax().setName("prob")) // TODO SoftMax is totally different with Caffe.

    Tools.compare(prototxt, bigdl, inputShape, outputShape)
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

    val conv1 = ConvolutionDnn(nInput, nOutput, kernel, kernel, stride, stride, pad, pad, 1)
      .setName("conv1")
    val conv2 = ConvolutionDnn(nInput, nOutput, kernel, kernel, stride, stride, pad, pad, 1)
      .setName("conv2")
    val model = nn.Sequential()
      .add(ConcatTableDnn().add(conv2).add(conv1))
      .add(CAddTableDnn().setName("eltwise"))

    Tools.compare(prototxt, model, inputShape, outputShape)
  }

  private def shape2Dim(shape: Array[Int]): String = {
    shape.map(x => "dim: " + x).mkString(" ")
  }
}
