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

import com.intel.analytics.bigdl.tensor.MklDnnTensor
import org.scalatest.{FlatSpec, Matchers}

class SingleLayer extends FlatSpec with Matchers {
  "relu" should "work correctly" in {
    val (batchSize, channel, height, width) = (4, 64, 112, 112)
    val prototxt =
      s"""
         |name: "relu-simple"
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
         |    shape: { dim: $batchSize dim: $channel dim: $height dim: $width }
         |  }
         |}
         |
         |layer {
         |  bottom: "data"
         |  top: "relu"
         |  name: "relu"
         |  type: "ReLU"
         |  relu_param {
         |  }
         |}
       """.stripMargin

    val identity = Collect.run(prototxt)

    val input = Tools.getTensor("Fwrd_data", Array(32, 64, 112, 112), identity)
    val output = Tools.getTensor("Fwrd_relu", Array(32, 64, 112, 112), identity)
    val gradOutput = Tools.getTensor("Bwrd_relu.loss", Array(32, 64, 112, 112), identity)
    val gradInput = Tools.getTensor("Bwrd_relu", Array(32, 64, 112, 112), identity)

    val relu = ReLUDnn[Float]()
    relu.forward(input)
    relu.backward(input, gradOutput)

    println()
  }

  "convolution" should "work correctly" in {
    val prototxt =
      s"""
         |name: "conv-simple"
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
         |    shape: { dim: 4 dim: 3 dim: 224 dim: 224 }
         |  }
         |}
         |
         |layer {
         |  bottom: "data"
         |  top: "conv"
         |  name: "conv"
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
       """.stripMargin

    val identity = Collect.run(prototxt)

    val input = Tools.getTensor("Fwrd_data", Array(4, 3, 224, 224), identity)
    val output = Tools.getTensor("Fwrd_conv", Array(4, 64, 112, 112), identity)
    val weight = Tools.getTensor("Fwrd_conv.Wght.0", Array(64, 3, 7, 7), identity)
    val bias = Tools.getTensor("Fwrd_conv.Wght.1", Array(64), identity)
    val gradOutput = Tools.getTensor("Bwrd_conv.loss", Array(4, 64, 112, 112), identity)
    val gradInput = Tools.getTensor("Bwrd_conv", Array(4, 3, 224, 224), identity)

    val conv = ConvolutionDnn(3, 64, 7, 7, 2, 2, 3, 3, 1, initWeight = weight, initBias = bias)

    val convOutput = {
      conv.forward(input)
      val reorder = MemoryReOrder(conv.output.asInstanceOf[MklDnnTensor[Float]].getFormat())
      reorder.forward(conv.output)
    }

    val convGradInput = {
      conv.backward(input, gradOutput)
      val reorder = MemoryReOrder(conv.gradInput.asInstanceOf[MklDnnTensor[Float]].getFormat())
      reorder.forward(conv.gradInput)
    }

    println()
  }
}
