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

class BatchNormalizationSpec extends FlatSpec with Matchers {
  def getPrototxt(test: TestCase): String = {
    val prototxt =
      s"""
         |name: "BatchNorm"
         |force_backward: true
         |layer {
         |  name: "data"
         |  type: "DummyData"
         |  top: "data"
         |  dummy_data_param {
         |    shape: {
         |      dim: ${test.batchSize}
         |      dim: ${test.channel}
         |      dim: ${test.height}
         |      dim: ${test.width}
         |    }
         |    data_filler {
         |      type: "gaussian"
         |      std: 0.01
         |    }
         |  }
         |}
         |
         |layer {
         |  name: "bn"
         |  type: "BatchNorm"
         |  bottom: "data"
         |  top: "bn"
         |
         |  batch_norm_param {
         |    eps: ${test.eps}
         |
         |    use_weight_bias: true
         |    bias_term: true
         |
         |    filler {
         |      type: "gaussian"
         |      std: 0.01
         |    }
         |
         |    bias_filler {
         |      type: "gaussian"
         |      std: 0.01
         |    }
         |
         |    engine: MKL2017
         |  }
         |}
         |
       """.stripMargin

    prototxt
  }

  def getModel(test: TestCase): BatchNormalization[Float] = {
    new BatchNormalization[Float](test.channel, test.eps)
  }

  def testLayer(prototxt: String, model: BatchNormalization[Float], test: TestCase): Unit = {
    val restOfSentence = (s"with parameters ${test.batchSize}", test.batchSize, test.channel,
      test.height, test.width, test.eps).productIterator.mkString(",")

    "A BatchNormalization" should restOfSentence in {
      val identity = Collect.run(prototxt)

      val input = Tools.loadTensor[Float]("Fwrd_data", Array(test.batchSize, test.channel,
        test.width, test.height), identity)
      val weights = Tools.loadTensor[Float]("Fwrd_bn.Wght.0", model.weight.size(), identity)
      val bias = Tools.loadTensor[Float]("Fwrd_bn.Wght.1", Array(test.channel), identity)

      model.weight.set(weights)
      model.bias.set(bias)

      for (i <- 0 until Tools.randTimes()) {
        model.forward(input)
        val output = Tools.loadTensor[Float]("Fwrd_bn", model.output.size(), identity)
        Tools.cumulativeError(model.output, output, "output") should be(0.0)

        val gradOutput = Tools.loadTensor[Float]("Bwrd_bn.loss", output.size(), identity)
        val gradInput = Tools.loadTensor[Float]("Bwrd_bn", input.size(), identity)

        model.zeroGradParameters()
        model.backward(input, gradOutput)

        val gradWeight = Tools.loadTensor[Float]("Bwrd_bn.Grad.0", weights.size(), identity)
        val gradBias = Tools.loadTensor[Float]("Bwrd_bn.Grad.1", bias.size(), identity)

        Tools.cumulativeError(model.weight, weights, "weights") should be (0.0)
        Tools.cumulativeError(model.bias, bias, "bias") should be (0.0)

        Tools.cumulativeError(model.output, output, "output") should be(0.0)
        Tools.cumulativeError(model.gradInput, gradInput, "gradient input") should be(0.0)
        Tools.cumulativeError(model.gradWeight, gradWeight, "gradWeight") should be(0.0)
        Tools.cumulativeError(model.gradBias, gradBias, "gradBias") should be(0.0)
      }
    }
  }

  def diffAttributes(): Unit = {
    val testCases = List(
      // VggLike
      TestCase(128, 128, 16, 16, 0.001),
      TestCase(128, 256, 8, 8, 0.001),
      TestCase(128, 512, 1, 1, 1.0E-5),
      TestCase(128, 512, 2, 2, 0.001),
      TestCase(128, 512, 4, 4, 0.001),
      TestCase(128, 64, 32, 32, 0.001),

      // GoogleNet v2
      TestCase(128, 128, 14, 14, 0.001),
      TestCase(128, 128, 2, 2, 0.001),
      TestCase(128, 128, 28, 28, 0.001),
      TestCase(128, 128, 4, 4, 0.001),
      TestCase(128, 128, 7, 7, 0.001),
      TestCase(128, 160, 14, 14, 0.001),
      TestCase(128, 160, 7, 7, 0.001),
      TestCase(128, 192, 14, 14, 0.001),
      TestCase(128, 192, 56, 56, 0.001),
      TestCase(128, 192, 7, 7, 0.001),
      TestCase(128, 224, 14, 14, 0.001),
      TestCase(128, 224, 7, 7, 0.001),
      TestCase(128, 256, 14, 14, 0.001),
      TestCase(128, 256, 7, 7, 0.001),
      TestCase(128, 320, 7, 7, 0.001),
      TestCase(128, 32, 28, 28, 0.001),
      TestCase(128, 352, 7, 7, 0.001),
      TestCase(128, 64, 112, 112, 0.001),
      TestCase(128, 64, 14, 14, 0.001),
      TestCase(128, 64, 28, 28, 0.001),
      TestCase(128, 64, 56, 56, 0.001),
      TestCase(128, 96, 14, 14, 0.001),
      TestCase(4, 96, 28, 28, 0.001)
    )

    for (test <- testCases) {
      val prototxt = getPrototxt(test)
      val model = getModel(test)
      
      testLayer(prototxt, model, test)
    }
  }

  def diffBatchSizeOnSameModel(): Unit = {
    val initTest = TestCase(128, 64, 14, 14, 0.001)
    val model = getModel(initTest)

    for (batchSize <- List(256, 372, 1002)) {
      val test = TestCase(batchSize, 64, 14, 14, 0.001)
      val prototxt = getPrototxt(test)

      testLayer(prototxt, model, test)
    }
  }

  def run(): Unit = {
    diffAttributes()
    diffBatchSizeOnSameModel()
  }

  run()

  case class TestCase(batchSize: Int, channel: Int, height: Int, width: Int, eps: Double)
}
