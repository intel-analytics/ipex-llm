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

import com.intel.analytics.bigdl.tensor.{MklDnnType, Tensor}
import org.scalatest.{FlatSpec, Matchers}

class SingleLayerSpec extends FlatSpec with Matchers {
  "convolution" should "work correctly" in {
    val inputShape = Array(4, 3, 5, 5)
    val outputShape = Array(4, 2, 3, 3)
    val name = "conv"
    val nOutput = 2
    val kernel = 3
    val pad = 1
    val stride = 2

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
         |    shape: { ${shape2Dim(inputShape)} }
         |  }
         |}
         |
         |layer {
         |  bottom: "data"
         |  top: "conv"
         |  name: "$name"
         |  type: "Convolution"
         |  convolution_param {
         |    num_output: $nOutput
         |    kernel_size: $kernel
         |    pad: $pad
         |    stride: $stride
         |    weight_filler {
         |      type: "msra"
         |      variance_norm: FAN_OUT
         |    }
         |    bias_filler {
         |      type: "gaussian"
         |    }
         |  }
         |}
       """.stripMargin

    val conv = ConvolutionDnn(3, nOutput, kernel, kernel, stride, stride, pad, pad, 1).setName(name)
    Tools.compare(prototxt, conv, inputShape, outputShape)
  }

  "convolution2" should "work correctly" in {
    val inputShape = Array(4, 3, 224, 224)
    val outputShape = Array(4, 64, 112, 112)
    val name = "conv"
    val nOutput = 64
    val kernel = 7
    val pad = 3
    val stride = 2

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
         |    shape: { ${shape2Dim(inputShape)} }
         |  }
         |}
         |
         |layer {
         |  bottom: "data"
         |  top: "conv"
         |  name: "$name"
         |  type: "Convolution"
         |  convolution_param {
         |    num_output: $nOutput
         |    kernel_size: $kernel
         |    pad: $pad
         |    stride: $stride
         |    weight_filler {
         |      type: "msra"
         |      variance_norm: FAN_OUT
         |    }
         |    bias_filler {
         |      type: "gaussian"
         |    }
         |  }
         |}
       """.stripMargin

    val conv = ConvolutionDnn(3, nOutput, kernel, kernel, stride, stride, pad, pad, 1).setName(name)
    Tools.compare(prototxt, conv, inputShape, outputShape)
  }

  "batch norm" should "work correctly" in {
    val (batchSize, channel, height, width) = (4, 64, 112, 112)
    val shape = Array(batchSize, channel, height, width)
    val prototxt = s"""
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
         |  top: "bn"
         |  name: "bn"
         |  type: "BatchNorm"
         |
         |  batch_norm_param {
         |    moving_average_fraction: 1.0
         |    filler { value: 1 }
         |    bias_filler { value: 0 }
         |    relu: false
         |    eps: 0.0
         |  }
         |}
       """.stripMargin

    val identity = Collect.run(prototxt)

    val input = Tools.getTensor("Fwrd_data", shape, identity)
    val output = Tools.getTensor("Fwrd_bn", shape, identity)
    val weight = Tools.getTensor("Fwrd_bn.Wght.3", Array(channel), identity)
    val bias = Tools.getTensor("Fwrd_bn.Wght.4", Array(channel), identity)
    val scale = Tools.getTensor("Fwrd_bn.Wght.2", Array(1), identity)
    val runningMean = Tools.getTensor("Fwrd_bn.Wght.0", Array(channel), identity)
    val runningVariance = Tools.getTensor("Fwrd_bn.Wght.1", Array(channel), identity)
    val gradOutput = Tools.getTensor("Bwrd_bn.loss", shape, identity)
    val gradInput = Tools.getTensor("Bwrd_bn", shape, identity)
    val gradWeight = Tools.getTensor("Bwrd_bn.Grad.3", Array(channel), identity)
    val gradBias = Tools.getTensor("Bwrd_bn.Grad.4", Array(channel), identity)
    val gradient = Tensor[Float](Array(2, channel))
    gradient.select(1, 1).copy(gradWeight)
    gradient.select(1, 2).copy(gradBias)

    val bn = SpatialBatchNormalization[Float](channel, eps = 0.0, momentum = 1.0, affine = true,
      initWeight = weight, initBias = bias).setShouldConvert(true)

    bn.forward(input)
    bn.backward(input, gradOutput)

    compare(weight, bn.weight)
    compare(bias, bn.bias)
    Tools.compare2Tensors(bn.output, output)
    Tools.compare2Tensors(runningMean, bn.runningMean)
    Tools.compare2Tensors(runningVariance, bn.runningVar)
    Tools.compare2Tensors(bn.diffAll, gradient.view(Array(2 * channel)))
    Tools.compare2Tensors(bn.gradInput, gradInput)
  }

  "max pooling" should "work correctly" in {
    val inputShape = Array(4, 64, 112, 112)
    val outputShape = Array(4, 64, 56, 56)
    val name = "pool"
    val prototxt =
      s"""
         |name: "maxpool-simple"
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
         |    shape: { ${shape2Dim(inputShape)} }
         |  }
         |}
         |
         |layer {
         |  bottom: "data"
         |  top: "pool"
         |  name: "$name"
         |  type: "Pooling"
         |  pooling_param {
         |    kernel_size: 3
         |    stride: 2
         |    pool: MAX
         |  }
         |}
       """.stripMargin

    val maxPooling = PoolingDnn[Float](3, 3, 2, 2).ceil().setName(name)

    Tools.compare(prototxt, maxPooling, inputShape, outputShape)
  }

  "avg pooling" should "work correctly" in {
    val inputShape = Array(4, 3, 7, 7)
    val outputShape = Array(4, 3, 3, 3)
    val name = "pool"
    val prototxt =
      s"""
         |name: "maxpool-simple"
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
         |    shape: { ${shape2Dim(inputShape)} }
         |  }
         |}
         |
         |layer {
         |  bottom: "data"
         |  top: "pool"
         |  name: "$name"
         |  type: "Pooling"
         |  pooling_param {
         |    kernel_size: 3
         |    stride: 2
         |    pool: AVE
         |  }
         |}
       """.stripMargin

    val avgPooling = PoolingDnnAverage[Float](3, 3, 2, 2).ceil().setName(name)
    Tools.compare(prototxt, avgPooling, inputShape, outputShape)
  }

  "linear " should "work correctly" in {
    val (batchSize, nInput) = (4, 64)
    val inputShape = Array(batchSize, nInput)
    val nOutput = 1000
    val outputShape = Array(batchSize, nOutput)
    val name = "fc"

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
         |    shape: { ${shape2Dim(inputShape)} }
         |  }
         |}
         |
         |layer {
         |  bottom: "data"
         |  top: "$name"
         |  name: "$name"
         |  type: "InnerProduct"
         |  inner_product_param {
         |    num_output: $nOutput
         |    weight_filler {
         |      type: "gaussian"
         |      std: 0.01
         |    }
         |    bias_filler {
         |      type: "constant"
         |      value: 0
         |    }
         |  }
         |}
       """.stripMargin
    val linear = Linear[Float](nInput, nOutput).setName(name)

    Tools.compare(prototxt, linear, inputShape, outputShape)
  }

  "relu" should "work correctly" in {
    val (batchSize, channel, height, width) = (4, 64, 112, 112)
    val inputShape = Array(batchSize, channel, height, width)
    val outputShape = inputShape
    val name = "relu"
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
         |  name: "$name"
         |  type: "ReLU"
         |  relu_param {
         |  }
         |}
       """.stripMargin

    val relu = ReLUDnn[Float]().setName(name)
    Tools.compare(prototxt, relu, inputShape, outputShape)
  }

  private def compare(src: Tensor[Float], dst: Tensor[Float]): Unit = {
    // todo the sync should be deleted.
    for (i <- List(src, dst)) {
      if (i.getTensorType == MklDnnType) {
        i.storage()
      }
    }

    src should be (dst)
  }

  private def shape2Dim(shape: Array[Int]): String = {
    shape.map(x => "dim: " + x).mkString(" ")
  }
}

