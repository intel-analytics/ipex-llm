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

import org.scalatest.{FlatSpec, Matchers}

class SpatialConvolutionSpec extends FlatSpec with Matchers {
  // must acquire half of all cores.
  Affinity.acquireCore()

  val testCases = List(
    // TODO current version of mkl dnn can't work on this testcase.
    // TestCase(512, 512, 3, 3, 1, 1, 1, 1, 1, 2, 2, 2048),

    // AlexNet
    TestCase(3, 96, 11, 11, 4, 4, 0, 0, 1, 227, 227, 32),
    TestCase(96, 256, 5, 5, 1, 1, 2, 2, 1, 27, 27, 32),
    TestCase(256, 384, 3, 3, 1, 1, 1, 1, 1, 13, 13, 32),
    TestCase(384, 384, 3, 3, 1, 1, 1, 1, 1, 13, 13, 32),
    TestCase(384, 256, 3, 3, 1, 1, 1, 1, 1, 13, 13, 32),

    // With 2 groups
    TestCase(96, 256, 5, 5, 1, 1, 2, 2, 2, 27, 27, 32),
    TestCase(384, 384, 3, 3, 1, 1, 1, 1, 2, 13, 13, 32),
    TestCase(384, 256, 3, 3, 1, 1, 1, 1, 2, 13, 13, 32),

    // GoogleNet v1
    TestCase(3, 64, 7, 7, 2, 2, 3, 3, 1, 224, 224, 32),
    TestCase(64, 64, 1, 1, 1, 1, 0, 0, 1, 56, 56, 32),
    TestCase(64, 192, 3, 3, 1, 1, 1, 1, 1, 56, 56, 32),
    TestCase(192, 64, 1, 1, 1, 1, 0, 0, 1, 28, 28, 32),
    TestCase(192, 96, 1, 1, 1, 1, 0, 0, 1, 28, 28, 32),
    TestCase(96, 128, 3, 3, 1, 1, 1, 1, 1, 28, 28, 32),
    TestCase(192, 16, 1, 1, 1, 1, 0, 0, 1, 28, 28, 32),
    TestCase(16, 32, 5, 5, 1, 1, 2, 2, 1, 28, 28, 32),
    TestCase(192, 32, 1, 1, 1, 1, 0, 0, 1, 28, 28, 32),
    TestCase(256, 128, 1, 1, 1, 1, 0, 0, 1, 28, 28, 32),
    TestCase(128, 192, 3, 3, 1, 1, 1, 1, 1, 28, 28, 32),
    TestCase(256, 32, 1, 1, 1, 1, 0, 0, 1, 28, 28, 32),
    TestCase(32, 96, 5, 5, 1, 1, 2, 2, 1, 28, 28, 32),
    TestCase(256, 64, 1, 1, 1, 1, 0, 0, 1, 28, 28, 32),
    TestCase(480, 192, 1, 1, 1, 1, 0, 0, 1, 14, 14, 32),
    TestCase(480, 96, 1, 1, 1, 1, 0, 0, 1, 14, 14, 32),
    TestCase(96, 208, 3, 3, 1, 1, 1, 1, 1, 14, 14, 32),
    TestCase(480, 16, 1, 1, 1, 1, 0, 0, 1, 14, 14, 32),
    TestCase(16, 16, 5, 5, 1, 1, 2, 2, 1, 14, 14, 32),
    TestCase(16, 48, 5, 5, 1, 1, 2, 2, 1, 14, 14, 32),
    TestCase(480, 64, 1, 1, 1, 1, 0, 0, 1, 14, 14, 32),
    TestCase(512, 160, 1, 1, 1, 1, 0, 0, 1, 14, 14, 32),
    TestCase(512, 112, 1, 1, 1, 1, 0, 0, 1, 14, 14, 32),
    TestCase(112, 224, 3, 3, 1, 1, 1, 1, 1, 14, 14, 32),
    TestCase(512, 24, 1, 1, 1, 1, 0, 0, 1, 14, 14, 32),
    TestCase(24, 64, 5, 5, 1, 1, 2, 2, 1, 14, 14, 32),
    TestCase(512, 64, 1, 1, 1, 1, 0, 0, 1, 14, 14, 32),
    TestCase(512, 128, 1, 1, 1, 1, 0, 0, 1, 14, 14, 32),
    TestCase(128, 256, 3, 3, 1, 1, 1, 1, 1, 14, 14, 32),
    TestCase(512, 144, 1, 1, 1, 1, 0, 0, 1, 14, 14, 32),
    TestCase(144, 288, 3, 3, 1, 1, 1, 1, 1, 14, 14, 32),
    TestCase(512, 32, 1, 1, 1, 1, 0, 0, 1, 14, 14, 32),
    TestCase(32, 64, 5, 5, 1, 1, 2, 2, 1, 14, 14, 32),
    TestCase(528, 256, 1, 1, 1, 1, 0, 0, 1, 14, 14, 32),
    TestCase(528, 160, 1, 1, 1, 1, 0, 0, 1, 14, 14, 32),
    TestCase(160, 320, 3, 3, 1, 1, 1, 1, 1, 14, 14, 32),
    TestCase(528, 32, 1, 1, 1, 1, 0, 0, 1, 14, 14, 32),
    TestCase(32, 128, 5, 5, 1, 1, 2, 2, 1, 14, 14, 32),
    TestCase(528, 128, 1, 1, 1, 1, 0, 0, 1, 14, 14, 32),
    TestCase(832, 256, 1, 1, 1, 1, 0, 0, 1, 7, 7, 32),
    TestCase(832, 160, 1, 1, 1, 1, 0, 0, 1, 7, 7, 32),
    TestCase(832, 32, 1, 1, 1, 1, 0, 0, 1, 7, 7, 32),
    TestCase(832, 128, 1, 1, 1, 1, 0, 0, 1, 7, 7, 32),
    TestCase(832, 384, 1, 1, 1, 1, 0, 0, 1, 7, 7, 32),
    TestCase(832, 192, 1, 1, 1, 1, 0, 0, 1, 7, 7, 32),
    TestCase(192, 384, 3, 3, 1, 1, 1, 1, 1, 7, 7, 32),
    TestCase(832, 48, 1, 1, 1, 1, 0, 0, 1, 7, 7, 32),
    TestCase(48, 128, 5, 5, 1, 1, 2, 2, 1, 7, 7, 32),
    TestCase(512, 128, 1, 1, 1, 1, 0, 0, 1, 4, 4, 32),

    // GoogleNet v2
    TestCase(64, 64, 3, 3, 1, 1, 1, 1, 1, 28, 28, 32),
    TestCase(64, 96, 3, 3, 1, 1, 1, 1, 1, 28, 28, 32),
    TestCase(96, 96, 3, 3, 1, 1, 1, 1, 1, 28, 28, 32),
    TestCase(320, 128, 1, 1, 1, 1, 0, 0, 1, 28, 28, 32),
    TestCase(128, 160, 3, 3, 2, 2, 1, 1, 1, 28, 28, 32),
    TestCase(320, 64, 1, 1, 1, 1, 0, 0, 1, 28, 28, 32),
    TestCase(96, 96, 3, 3, 2, 2, 1, 1, 1, 28, 28, 32),
    TestCase(576, 224, 1, 1, 1, 1, 0, 0, 1, 14, 14, 32),
    TestCase(576, 64, 1, 1, 1, 1, 0, 0, 1, 14, 14, 32),
    TestCase(576, 128, 1, 1, 1, 1, 0, 0, 1, 14, 14, 32),
    TestCase(576, 192, 1, 1, 1, 1, 0, 0, 1, 14, 14, 32),
    TestCase(576, 96, 1, 1, 1, 1, 0, 0, 1, 14, 14, 32),
    TestCase(96, 128, 3, 3, 1, 1, 1, 1, 1, 14, 14, 32),
    TestCase(128, 128, 3, 3, 1, 1, 1, 1, 1, 14, 14, 32),
    TestCase(576, 160, 1, 1, 1, 1, 0, 0, 1, 14, 14, 32),
    TestCase(128, 160, 3, 3, 1, 1, 1, 1, 1, 14, 14, 32),
    TestCase(160, 160, 3, 3, 1, 1, 1, 1, 1, 14, 14, 32),
    TestCase(128, 192, 3, 3, 1, 1, 1, 1, 1, 14, 14, 32),
    TestCase(160, 192, 3, 3, 1, 1, 1, 1, 1, 14, 14, 32),
    TestCase(192, 192, 3, 3, 1, 1, 1, 1, 1, 14, 14, 32),
    TestCase(128, 192, 3, 3, 2, 2, 1, 1, 1, 14, 14, 32),
    TestCase(192, 256, 3, 3, 1, 1, 1, 1, 1, 14, 14, 32),
    TestCase(256, 256, 3, 3, 2, 2, 1, 1, 1, 14, 14, 32),
    TestCase(192, 320, 3, 3, 1, 1, 1, 1, 1, 7, 7, 32),
    TestCase(1024, 160, 1, 1, 1, 1, 0, 0, 1, 7, 7, 32),
    TestCase(160, 224, 3, 3, 1, 1, 1, 1, 1, 7, 7, 32),
    TestCase(224, 224, 3, 3, 1, 1, 1, 1, 1, 7, 7, 32),
    TestCase(1024, 128, 1, 1, 1, 1, 0, 0, 1, 7, 7, 32),
    TestCase(1024, 352, 1, 1, 1, 1, 0, 0, 1, 7, 7, 32),
    TestCase(1024, 192, 1, 1, 1, 1, 0, 0, 1, 7, 7, 32),
    TestCase(192, 224, 3, 3, 1, 1, 1, 1, 1, 7, 7, 32),
    TestCase(1024, 128, 1, 1, 1, 1, 0, 0, 1, 2, 2, 32),
    TestCase(576, 128, 1, 1, 1, 1, 0, 0, 1, 4, 4, 32),

    // VggLike
    TestCase(3, 64, 3, 3, 1, 1, 1, 1, 1, 32, 32, 128),
    TestCase(64, 64, 3, 3, 1, 1, 1, 1, 1, 32, 32, 128),
    TestCase(64, 128, 3, 3, 1, 1, 1, 1, 1, 16, 16, 128),
    TestCase(128, 128, 3, 3, 1, 1, 1, 1, 1, 16, 16, 128)
  )

  for (test <- testCases) {
    val restOfSentence = (s"with parameters ${test.nInputPlane}", test.nOutputPlane,
      test.kW, test.kH, test.dW, test.dH, test.padW, test.padH,
      test.inputWidth, test.inputHeight, test.group).productIterator.mkString(",")

    "A SpatialConvolution" should restOfSentence in {
      val prototxt =
        s"""
           |name: "Convolution"
           |force_backward: true
           |layer {
           |  name: "data"
           |  type: "DummyData"
           |  top: "data"
           |  dummy_data_param {
           |    shape: {
           |      dim: ${test.batchSize}
           |      dim: ${test.nInputPlane}
           |      dim: ${test.inputHeight}
           |      dim: ${test.inputWidth}
           |    }
           |    data_filler {
           |      type: "xavier"
           |      std: 0.01
           |    }
           |  }
           |}
           |
           |layer {
           |  name: "conv"
           |  type: "Convolution"
           |  bottom: "data"
           |  top: "conv"
           |
           |  convolution_param {
           |    num_output: ${test.nOutputPlane}
           |
           |    kernel_h: ${test.kH}
           |    kernel_w: ${test.kW}
           |    stride_h: ${test.dH}
           |    stride_w: ${test.dW}
           |    pad_h: ${test.padH}
           |    pad_w: ${test.padW}
           |
           |    group: ${test.group}
           |
           |    weight_filler {
           |      type: "xavier"
           |    }
           |    bias_filler {
           |      type: "xavier"
           |    }
           |    engine: MKL2017
           |  }
           |}
        """.stripMargin
      val identity = Collect.run(prototxt)

      val model = new SpatialConvolution[Float](test.nInputPlane, test.nOutputPlane,
        test.kW, test.kH, test.dW, test.dH, test.padW, test.padH, test.group)

      val input = Tools.loadTensor[Float]("Fwrd_data", Array(test.batchSize, test.nInputPlane,
                                                      test.inputWidth, test.inputHeight), identity)
      val weights = Tools.loadTensor[Float]("Fwrd_conv.Wght.0", model.weight.size(), identity)
      val bias = Tools.loadTensor[Float]("Fwrd_conv.Wght.1", Array(test.nOutputPlane), identity)

      model.weight.set(weights)
      model.bias.set(bias)

      model.forward(input)

      val output = Tools.loadTensor[Float]("Fwrd_conv", model.output.size(), identity)
      Tools.cumulativeError(model.output, output, "output") should be(0.0)

      val gradOutput = Tools.loadTensor[Float]("Bwrd_conv.loss", model.output.size(), identity)
      val gradInput = Tools.loadTensor[Float]("Bwrd_conv", input.size(), identity)
      val gradWeight = Tools.loadTensor[Float]("Bwrd_conv.Grad.0", weights.size(), identity)
      val gradBias = Tools.loadTensor[Float]("Bwrd_conv.Grad.1", bias.size(), identity)

      model.forward(input)
      model.zeroGradParameters()
      model.backward(input, gradOutput)
      model.backward(input, gradOutput)

      Tools.cumulativeError(model.gradInput, gradInput, "gradient input") should be(0.0)
      Tools.cumulativeError(model.gradWeight, gradWeight, "gradWeight") should be(0.0)
      Tools.cumulativeError(model.gradBias, gradBias, "gradBias") should be(0.0)
    }
  }

  case class TestCase(nInputPlane : Int, nOutputPlane : Int, kW : Int, kH : Int,
                      dW : Int, dH : Int, padW : Int, padH : Int, group: Int,
                      inputWidth : Int, inputHeight : Int, batchSize : Int)
}
