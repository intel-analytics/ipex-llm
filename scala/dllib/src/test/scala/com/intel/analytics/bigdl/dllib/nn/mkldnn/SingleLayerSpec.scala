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

import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.nn.mkldnn.Phase.TrainingPhase
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class SingleLayerSpec extends FlatSpec with Matchers with BeforeAndAfter {
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

    val conv = SpatialConvolution(3, nOutput, kernel, kernel, stride, stride, pad, pad, 1)
      .setName(name)
    val seq = Sequential()
        .add(Input(inputShape, Memory.Format.nchw))
        .add(conv)
        .add(ReorderMemory(HeapData(outputShape, Memory.Format.nchw)))
    seq.compile(TrainingPhase)

    Tools.compare(prototxt, seq, inputShape, outputShape, 1e-6)
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

    val conv = SpatialConvolution(3, nOutput, kernel, kernel, stride, stride, pad, pad, 1)
      .setName(name)
    val seq = Sequential()
      .add(Input(inputShape, Memory.Format.nchw))
      .add(conv)
      .add(ReorderMemory(HeapData(outputShape, Memory.Format.nchw)))

    seq.compile(TrainingPhase, Array(HeapData(inputShape, Memory.Format.nchw)))

    Tools.compare(prototxt, seq, inputShape, outputShape)
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

    val maxPooling = MaxPooling(3, 3, 2, 2).setName(name)
    maxPooling.setRuntime(new MklDnnRuntime)
    maxPooling.initFwdPrimitives(Array(HeapData(inputShape, Memory.Format.nchw)), TrainingPhase)
    maxPooling.initBwdPrimitives(Array(HeapData(outputShape, Memory.Format.nchw)), TrainingPhase)

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

    val avgPooling = AvgPooling(3, 3, 2, 2).setName(name)
    avgPooling.setRuntime(new MklDnnRuntime)
    avgPooling.initFwdPrimitives(Array(HeapData(inputShape, Memory.Format.nchw)), TrainingPhase)
    avgPooling.initBwdPrimitives(Array(HeapData(outputShape, Memory.Format.nchw)), TrainingPhase)
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
    val linear = Linear(nInput, nOutput).setName(name)
    linear.setRuntime(new MklDnnRuntime)
    linear.initFwdPrimitives(Array(HeapData(inputShape, Memory.Format.nc)), TrainingPhase)
    linear.initBwdPrimitives(Array(HeapData(outputShape, Memory.Format.nc)), TrainingPhase)
    linear.initGradWPrimitives(Array(HeapData(outputShape, Memory.Format.nc)), TrainingPhase)

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

    val relu = ReLU().setName(name)
    relu.setRuntime(new MklDnnRuntime)
    relu.initFwdPrimitives(Array(HeapData(inputShape, Memory.Format.nchw)), TrainingPhase)
    relu.initBwdPrimitives(Array(HeapData(outputShape, Memory.Format.nchw)), TrainingPhase)
    Tools.compare(prototxt, relu, inputShape, outputShape)
  }

  private def shape2Dim(shape: Array[Int]): String = {
    shape.map(x => "dim: " + x).mkString(" ")
  }
}

