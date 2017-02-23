/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
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

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.dnn.Tools._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.collection.mutable.ArrayBuffer

object AlexNet {
  def apply(classNum: Int): Module[Float] = {
    val model = new Sequential()

    model.add(new SpatialConvolution(3, 96, 11, 11, 4, 4, propagateBack = true)
        .setName("conv1")
        .setInitMethod(Xavier))

    model.add(new ReLU(ip = false).setName("relu1"))
    model.add(new SpatialCrossMapLRN(5, 0.0001, 0.75).setName("norm1"))
    model.add(new SpatialMaxPooling(3, 3, 2, 2).setName("pool1"))
    model.add(new SpatialConvolution(96, 256, 5, 5, 1, 1, 2, 2, 2).setName("conv2"))
    model.add(new ReLU(ip = false).setName("relu2"))
    model.add(new SpatialCrossMapLRN(5, 0.0001, 0.75).setName("norm2"))
    model.add(new SpatialMaxPooling(3, 3, 2, 2).setName("pool2"))
    model.add(new SpatialConvolution(256, 384, 3, 3, 1, 1, 1, 1).setName("conv3"))
    model.add(new ReLU(ip = false).setName("relu3"))
    model.add(new SpatialConvolution(384, 384, 3, 3, 1, 1, 1, 1, 2).setName("conv4"))
    model.add(new ReLU(ip = false).setName("relu4"))
    model.add(new SpatialConvolution(384, 256, 3, 3, 1, 1, 1, 1, 2).setName("conv5"))
    model.add(new ReLU(ip = false).setName("relu5"))
    model.add(new SpatialMaxPooling(3, 3, 2, 2).setName("pool5"))
    model.add(new View(256 * 6 * 6))
    model.add(new Linear(256 * 6 * 6, 4096).setName("fc6"))
    model.add(new ReLU(ip = false).setName("relu6"))
    model.add(new Dropout(0.5).setName("drop6"))
    model.add(new Linear(4096, 4096).setName("fc7"))
    model.add(new ReLU(ip = false).setName("relu7"))
    model.add(new Dropout(0.5).setName("drop7"))
    model.add(new Linear(4096, classNum).setName("fc8"))

    /**
     * needn't have LogsOftMax, because the different implementation
     */
    // model.add(LogSoftMax().setName("loss"))
    model
  }
}

class AlexNetSpec extends FlatSpec with BeforeAndAfter with Matchers {
  before {
    Affinity.acquireCore()
  }

  "An AlexNet forward and backward" should "the same output, gradient as intelcaffe w/ dnn" in {
    val model = AlexNet(1000)
    model.convertToMklDnn()
    model.reset()

    val modules = ArrayBuffer[TensorModule[Float]]()
    flattenModules(model, modules)

    def iteration(): Unit = {
      val identity = Collect.run(alexnet, singleLayer = false)
      val input = loadTensor("Fwrd_data_input", Array(batchSize, 3, 227, 227), identity)

      loadParameters(modules, identity)

      val output = model.forward(input).toTensor
      val caffeOutput = loadTensor("Fwrd_fc8", output.size(), identity)

      compareAllLayers(modules, identity, Forward)
      cumulativeError(output, caffeOutput, "output") should be(0.0)

      val gradOutput = loadTensor("Bwrd_loss", output.size(), identity)
      val gradInput = model.backward(input, gradOutput).toTensor
      val gradInputCaffe = loadTensor("Bwrd_conv1", gradInput.size(), identity)

      compareAllLayers(modules, identity, Backward)
      compareParameters(modules, identity)
      cumulativeError(gradInput, gradInputCaffe, "gradInput") should be(0.0)
    }

    for (i <- 0 until 5) {
      iteration()
    }
  }

  val batchSize = 4
  val alexnet = s"""
name: "AlexNet"
force_backward: true
layer {
  name: "data_input"
  type: "DummyData"
  top: "data"
  include {
    phase: TRAIN
  }
  dummy_data_param {
    shape: { dim: $batchSize dim: 3 dim: 227 dim: 227 }
    data_filler {
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
      value: 0
    }
  }
}

layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 96
    kernel_size: 11
    stride: 4
    engine: MKL2017
    weight_filler {
#      type: "gaussian"
#      std: 0.01
      type: "constant"
      value: 0.1
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "relu1"
  type: "ReLU"
  bottom: "conv1"
  top: "conv1"
  relu_param {
    engine: MKL2017
  }
}
layer {
  name: "norm1"
  type: "LRN"
  bottom: "conv1"
  top: "norm1"
  lrn_param {
    local_size: 5
    alpha: 0.0001
    beta: 0.75
    k: 1.0
    engine: MKL2017
  }
}
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "norm1"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
    engine: MKL2017
  }
}
layer {
  name: "conv2"
  type: "Convolution"
  bottom: "pool1"
  top: "conv2"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 256
    pad: 2
    kernel_size: 5
    group: 2
    engine: MKL2017
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0.1
    }
  }
}
layer {
  name: "relu2"
  type: "ReLU"
  bottom: "conv2"
  top: "conv2"
  relu_param {
    engine: MKL2017
  }
}
layer {
  name: "norm2"
  type: "LRN"
  bottom: "conv2"
  top: "norm2"
  lrn_param {
    local_size: 5
    alpha: 0.0001
    beta: 0.75
    engine: MKL2017
  }
}
layer {
  name: "pool2"
  type: "Pooling"
  bottom: "norm2"
  top: "pool2"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
    engine: MKL2017
  }
}
layer {
  name: "conv3"
  type: "Convolution"
  bottom: "pool2"
  top: "conv3"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 384
    pad: 1
    kernel_size: 3
    engine: MKL2017
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "relu3"
  type: "ReLU"
  bottom: "conv3"
  top: "conv3"
  relu_param {
    engine: MKL2017
  }
}
layer {
  name: "conv4"
  type: "Convolution"
  bottom: "conv3"
  top: "conv4"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 384
    pad: 1
    kernel_size: 3
    group: 2
    engine: MKL2017
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0.1
    }
  }
}
layer {
  name: "relu4"
  type: "ReLU"
  bottom: "conv4"
  top: "conv4"
  relu_param {
    engine: MKL2017
  }
}
layer {
  name: "conv5"
  type: "Convolution"
  bottom: "conv4"
  top: "conv5"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 256
    pad: 1
    kernel_size: 3
    group: 2
    engine: MKL2017
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0.1
    }
  }
}
layer {
  name: "relu5"
  type: "ReLU"
  bottom: "conv5"
  top: "conv5"
  relu_param {
    engine: MKL2017
  }
}
layer {
  name: "pool5"
  type: "Pooling"
  bottom: "conv5"
  top: "pool5"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
    engine: MKL2017
  }
}
layer {
  name: "fc6"
  type: "InnerProduct"
  bottom: "pool5"
  top: "fc6"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  inner_product_param {
    num_output: 4096
    weight_filler {
      type: "gaussian"
      std: 0.005
    }
    bias_filler {
      type: "constant"
      value: 0.1
    }
  }
}
layer {
  name: "relu6"
  type: "ReLU"
  bottom: "fc6"
  top: "fc6"
  relu_param {
    engine: MKL2017
  }
}
#layer {
#  name: "drop6"
#  type: "Dropout"
#  bottom: "fc6"
#  top: "fc6"
#  dropout_param {
#    dropout_ratio: 0.5
#  }
#}
layer {
  name: "fc7"
  type: "InnerProduct"
  bottom: "fc6"
  top: "fc7"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  inner_product_param {
    num_output: 4096
    weight_filler {
      type: "gaussian"
      std: 0.005
    }
    bias_filler {
      type: "constant"
      value: 0.1
    }
  }
}
layer {
  name: "relu7"
  type: "ReLU"
  bottom: "fc7"
  top: "fc7"
  relu_param {
    engine: MKL2017
  }
}
#layer {
#  name: "drop7"
#  type: "Dropout"
#  bottom: "fc7"
#  top: "fc7"
#  dropout_param {
#    dropout_ratio: 0.5
#  }
#}
layer {
  name: "fc8"
  type: "InnerProduct"
  bottom: "fc7"
  top: "fc8"
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
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "accuracy"
  type: "Accuracy"
  bottom: "fc8"
  bottom: "label"
  top: "accuracy"
  include {
    phase: TEST
  }
}

layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "fc8"
  bottom: "label"
  top: "loss"
  loss_param {
    normalization: VALID
  }
}
      """
}
