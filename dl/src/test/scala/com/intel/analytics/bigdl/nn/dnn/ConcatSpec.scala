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
import com.intel.analytics.bigdl.nn.dnn.Tools._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.nn.abstractnn.TensorModule

import scala.collection.mutable.ArrayBuffer
import org.scalatest.{FlatSpec, Matchers}

class ConcatSpec extends FlatSpec with Matchers {
  "A Concat" should "aa" in {
    val prototxt =
      s"""
         |name: "Concat"
         |force_backward: true
         |
         |layer {
         |  name: "data"
         |  type: "DummyData"
         |  top: "data"
         |  dummy_data_param {
         |    shape: { dim: 32 dim: 192 dim: 28 dim: 28}
         |    data_filler {
         |      type: "gaussian"
         |      std: 0.01
         |    }
         |  }
         |}
         |layer {
         |  name: "split"
         |  type: "Split"
         |  split_param {
         |    engine: MKL2017
         |  }
         |  bottom: "data"
         |  top: "split1"
         |  top: "split2"
         |  top: "split3"
         |  top: "split4"
         |}
         |layer {
         |  name: "inception_3a/1x1"
         |  type: "Convolution"
         |  bottom: "split1"
         |  top: "inception_3a/1x1"
         |  param {
         |    lr_mult: 1
         |    decay_mult: 1
         |  }
         |  param {
         |    lr_mult: 2
         |    decay_mult: 0
         |  }
         |  convolution_param {
         |    engine: MKL2017
         |    num_output: 64
         |    kernel_size: 1
         |    weight_filler {
         |      type: "xavier"
         |    }
         |    bias_filler {
         |      type: "constant"
         |      value: 0.2
         |    }
         |  }
         |}
         |layer {
         |  name: "inception_3a/relu_1x1"
         |  type: "ReLU"
         |  relu_param {
         |    engine: MKL2017
         |  }
         |  bottom: "inception_3a/1x1"
         |  top: "inception_3a/1x1"
         |}
         |layer {
         |  name: "inception_3a/3x3_reduce"
         |  type: "Convolution"
         |  bottom: "split2"
         |  top: "inception_3a/3x3_reduce"
         |  param {
         |    lr_mult: 1
         |    decay_mult: 1
         |  }
         |  param {
         |    lr_mult: 2
         |    decay_mult: 0
         |  }
         |  convolution_param {
         |    engine: MKL2017
         |    num_output: 96
         |    kernel_size: 1
         |    weight_filler {
         |      type: "xavier"
         |    }
         |    bias_filler {
         |      type: "constant"
         |      value: 0.2
         |    }
         |  }
         |}
         |layer {
         |  name: "inception_3a/relu_3x3_reduce"
         |  type: "ReLU"
         |  relu_param {
         |    engine: MKL2017
         |  }
         |  bottom: "inception_3a/3x3_reduce"
         |  top: "inception_3a/3x3_reduce"
         |}
         |layer {
         |  name: "inception_3a/3x3"
         |  type: "Convolution"
         |  bottom: "inception_3a/3x3_reduce"
         |  top: "inception_3a/3x3"
         |  param {
         |    lr_mult: 1
         |    decay_mult: 1
         |  }
         |  param {
         |    lr_mult: 2
         |    decay_mult: 0
         |  }
         |  convolution_param {
         |    engine: MKL2017
         |    num_output: 128
         |    pad: 1
         |    kernel_size: 3
         |    weight_filler {
         |      type: "xavier"
         |    }
         |    bias_filler {
         |      type: "constant"
         |      value: 0.2
         |    }
         |  }
         |}
         |layer {
         |  name: "inception_3a/relu_3x3"
         |  type: "ReLU"
         |  relu_param {
         |    engine: MKL2017
         |  }
         |  bottom: "inception_3a/3x3"
         |  top: "inception_3a/3x3"
         |}
         |layer {
         |  name: "inception_3a/5x5_reduce"
         |  type: "Convolution"
         |  bottom: "split3"
         |  top: "inception_3a/5x5_reduce"
         |  param {
         |    lr_mult: 1
         |    decay_mult: 1
         |  }
         |  param {
         |    lr_mult: 2
         |    decay_mult: 0
         |  }
         |  convolution_param {
         |    engine: MKL2017
         |    num_output: 16
         |    kernel_size: 1
         |    weight_filler {
         |      type: "xavier"
         |    }
         |    bias_filler {
         |      type: "constant"
         |      value: 0.2
         |    }
         |  }
         |}
         |layer {
         |  name: "inception_3a/relu_5x5_reduce"
         |  type: "ReLU"
         |  relu_param {
         |    engine: MKL2017
         |  }
         |  bottom: "inception_3a/5x5_reduce"
         |  top: "inception_3a/5x5_reduce"
         |}
         |layer {
         |  name: "inception_3a/5x5"
         |  type: "Convolution"
         |  bottom: "inception_3a/5x5_reduce"
         |  top: "inception_3a/5x5"
         |  param {
         |    lr_mult: 1
         |    decay_mult: 1
         |  }
         |  param {
         |    lr_mult: 2
         |    decay_mult: 0
         |  }
         |  convolution_param {
         |    engine: MKL2017
         |    num_output: 32
         |    pad: 2
         |    kernel_size: 5
         |    weight_filler {
         |      type: "xavier"
         |    }
         |    bias_filler {
         |      type: "constant"
         |      value: 0.2
         |    }
         |  }
         |}
         |layer {
         |  name: "inception_3a/relu_5x5"
         |  type: "ReLU"
         |  relu_param {
         |    engine: MKL2017
         |  }
         |  bottom: "inception_3a/5x5"
         |  top: "inception_3a/5x5"
         |}
         |layer {
         |  name: "inception_3a/pool"
         |  type: "Pooling"
         |  bottom: "split4"
         |  top: "inception_3a/pool"
         |  pooling_param {
         |    engine: MKL2017
         |    pool: MAX
         |    kernel_size: 3
         |    stride: 1
         |    pad: 1
         |  }
         |}
         |layer {
         |  name: "inception_3a/pool_proj"
         |  type: "Convolution"
         |  bottom: "inception_3a/pool"
         |  top: "inception_3a/pool_proj"
         |  param {
         |    lr_mult: 1
         |    decay_mult: 1
         |  }
         |  param {
         |    lr_mult: 2
         |    decay_mult: 0
         |  }
         |  convolution_param {
         |    engine: MKL2017
         |    num_output: 32
         |    kernel_size: 1
         |    weight_filler {
         |      type: "xavier"
         |    }
         |    bias_filler {
         |      type: "constant"
         |      value: 0.2
         |    }
         |  }
         |}
         |layer {
         |  name: "inception_3a/relu_pool_proj"
         |  type: "ReLU"
         |  relu_param {
         |    engine: MKL2017
         |  }
         |  bottom: "inception_3a/pool_proj"
         |  top: "inception_3a/pool_proj"
         |}
         |layer {
         |  name: "inception_3a/output"
         |  type: "Concat"
         |  concat_param {
         |    engine: MKL2017
         |  }
         |  bottom: "inception_3a/1x1"
         |  bottom: "inception_3a/3x3"
         |  bottom: "inception_3a/5x5"
         |  bottom: "inception_3a/pool_proj"
         |  top: "inception_3a/output"
         |}
       """.stripMargin

    def getModel: Module[Float] = {
      val concat = new Concat(2)

      val conv1 = new Sequential()
      val conv3 = new Sequential()
      val conv5 = new Sequential()
      val pool = new Sequential()

      conv1.add(new SpatialConvolution(192, 64, 1, 1, 1, 1, 0, 0)
        .setInitMethod(Xavier).setName("inception_3a/1x1"))
      conv1.add(new ReLU(ip = false).setName("inception_3a/relu_1x1"))

      conv3.add(new SpatialConvolution(192, 96, 1, 1, 1, 1, 0, 0).setInitMethod(Xavier)
        .setName("inception_3a/3x3_reduce"))
      conv3.add(new ReLU(ip = false)
        .setName("inception_3a/relu_3x3_reduce"))
      conv3.add(new SpatialConvolution(96, 128, 3, 3, 1, 1, 1, 1).setInitMethod(Xavier)
        .setName("inception_3a/3x3"))
      conv3.add(new ReLU(ip = false)
        .setName("inception_3a/relu_3x3"))

      conv5.add(new SpatialConvolution(192, 16, 1, 1, 1, 1, 0, 0).setInitMethod(Xavier)
        .setName("inception_3a/5x5_reduce"))
      conv5.add(new ReLU(ip = false).setName("inception_3a/relu_5x5_reduce"))
      conv5.add(new SpatialConvolution(16, 32, 5, 5, 1, 1, 2, 2).setInitMethod(Xavier)
        .setName("inception_3a/5x5"))
      conv5.add(new ReLU(ip = false)
        .setName("inception_3a/relu_5x5"))

      pool.add(new SpatialMaxPooling(3, 3, 1, 1, 1, 1).ceil().setName("inception_3a/pool"))
      pool.add(new SpatialConvolution(192, 32, 1, 1, 1, 1, 0, 0).setInitMethod(Xavier)
        .setName("inception_3a/pool_proj"))
      pool.add(new ReLU(ip = false).setName("inception_3a/relu_pool_proj"))

      concat.add(conv1)
      concat.add(conv3)
      concat.add(conv5)
      concat.add(pool)
      concat
    }

    val identity = Collect.run(prototxt, singleLayer = true)

    val model = getModel
    model.convertToMklDnn()

    val modules = ArrayBuffer[TensorModule[Float]]()
    flattenModules(model, modules)

    loadParameters(modules, identity)

    val input = loadTensor("Fwrd_data", Array(4, 192, 28, 28), identity)
    val gradOutput = loadTensor("Bwrd_inception_3a_output.loss", Array(4, 256, 28, 28), identity)

    model.forward(input)
    model.backward(input, gradOutput)

    compareAllLayers(modules, identity, Forward)
    compareAllLayers(modules, identity, Backward)

    val modelOutput = model.output.toTensor
    val modelGradInput = model.gradInput.toTensor

    val output = loadTensor("Fwrd_inception_3a_output", modelOutput.size(), identity)
    val gradInput = loadTensor("Bwrd_split", input.size(), identity)

    cumulativeError(modelOutput, output, "output") should be(0.0)
    cumulativeError(modelGradInput, gradInput, "gradient input") should be(0.0)
  }
}
