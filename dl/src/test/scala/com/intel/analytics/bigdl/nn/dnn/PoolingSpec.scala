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

import com.intel.analytics.bigdl.nn.dnn.Tools._
import com.intel.analytics.bigdl.numeric.NumericFloat

import org.scalatest.{FlatSpec, Matchers}

class PoolingSpec extends FlatSpec with Matchers {
  def getModel(kW: Int, kH: Int, dW: Int, dH: Int,
               padW: Int, padH: Int, ver : String) : Pool[Float] = {
    ver match {
      case "MAX" =>
        new SpatialMaxPooling[Float](kW, kH, dW, dH, padW, padH)
      case "AVE" =>
        new SpatialAveragePooling[Float](kW, kH, dW, dH, padW, padH)
    }
  }

  def getModel(test: TestCase, ver : String) : Pool[Float] = {
    ver match {
      case "MAX" =>
        new SpatialMaxPooling[Float](test.kW, test.kH, test.dW, test.dH, test.padW, test.padH)
      case "AVE" =>
        new SpatialAveragePooling[Float](test.kW, test.kH, test.dW, test.dH, test.padW, test.padH)
    }
  }

  def getPrototxt(test: TestCase, poolType: String): String = {
    val prototxt =
      s"""
         |name: "Pooling"
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
         |  name: "pooling"
         |  type: "Pooling"
         |  bottom: "data"
         |  top: "pooling"
         |
         |  pooling_param {
         |    engine: MKL2017
         |    pool: $poolType
         |    kernel_h: ${test.kH}
         |    kernel_w: ${test.kW}
         |    stride_h: ${test.dH}
         |    stride_w: ${test.dW}
         |    pad_h: ${test.padH}
         |    pad_w: ${test.padW}
         |  }
         |}
         |
       """.stripMargin
    prototxt
  }

  def doTest(prototxt: String, model: Pool[Float], test: TestCase): Unit = {
    val identity = Collect.run(prototxt)

    val input = loadTensor("Fwrd_data", Array(test.batchSize, test.channel,
      test.width, test.height), identity)

    model.forward(input)
    val output = loadTensor("Fwrd_pooling", model.output.size(), identity)

    val gradOutput = loadTensor("Bwrd_pooling.loss", output.size(), identity)
    val gradInput = loadTensor("Bwrd_pooling", input.size(), identity)

    for (i <- 0 until randTimes()) {
      model.forward(input)
      model.zeroGradParameters()
      model.backward(input, gradOutput)
    }

    cumulativeError(model.output, output, "output") should be(0.0)
    cumulativeError(model.gradInput, gradInput, "gradient input") should be(0.0)
  }

  def diffAttributes(): Unit = {
    val testCases = List(
      TestCase(128, 128, 16, 16, 2, 2, 2, 2, 0, 0),
      TestCase(128, 256, 13, 13, 3, 3, 2, 2, 0, 0),
      TestCase(128, 256, 27, 27, 3, 3, 2, 2, 0, 0),
      TestCase(128, 256, 8, 8, 2, 2, 2, 2, 0, 0),
      TestCase(128, 512, 2, 2, 2, 2, 2, 2, 0, 0),
      TestCase(128, 512, 4, 4, 2, 2, 2, 2, 0, 0),
      TestCase(128, 64, 32, 32, 2, 2, 2, 2, 0, 0),
      TestCase(128, 96, 55, 55, 3, 3, 2, 2, 0, 0),
      TestCase(128, 1024, 7, 7, 3, 3, 1, 1, 1, 1),
      TestCase(128, 1024, 7, 7, 5, 5, 3, 3, 0, 0),
      TestCase(128, 1024, 7, 7, 7, 7, 1, 1, 0, 0),
      TestCase(128, 192, 28, 28, 3, 3, 1, 1, 1, 1),
      TestCase(128, 192, 56, 56, 3, 3, 2, 2, 0, 0),
      TestCase(128, 256, 28, 28, 3, 3, 1, 1, 1, 1),
      TestCase(128, 320, 28, 28, 3, 3, 2, 2, 0, 0),
      TestCase(128, 480, 14, 14, 3, 3, 1, 1, 1, 1),
      TestCase(128, 480, 28, 28, 3, 3, 2, 2, 0, 0),
      TestCase(128, 512, 14, 14, 3, 3, 1, 1, 1, 1),
      TestCase(128, 512, 14, 14, 5, 5, 3, 3, 0, 0),
      TestCase(128, 528, 14, 14, 3, 3, 1, 1, 1, 1),
      TestCase(128, 528, 14, 14, 5, 5, 3, 3, 0, 0),
      TestCase(128, 576, 14, 14, 3, 3, 1, 1, 1, 1),
      TestCase(128, 576, 14, 14, 3, 3, 2, 2, 0, 0),
      TestCase(128, 576, 14, 14, 5, 5, 3, 3, 0, 0),
      TestCase(128, 64, 112, 112, 3, 3, 2, 2, 0, 0),
      TestCase(128, 832, 14, 14, 3, 3, 2, 2, 0, 0),
      TestCase(128, 832, 7, 7, 3, 3, 1, 1, 1, 1)
    )

    for (test <- testCases) {
      val model = getModel(test.kW, test.kH, test.dW, test.dH, test.padW, test.padH, "MAX")
      val prototxt = getPrototxt(test, "MAX")

      val restOfSentence = (s"with parameters ${test.batchSize}", test.channel, test.height,
        test.width, test.kW, test.kH, test.dW, test.dH, test.padW, test.padH)
        .productIterator.mkString(",")

      "A MaxPooling" should restOfSentence in {
        doTest(prototxt, model, test)
      }
    }

    for (test <- testCases) {
      val model = getModel(test.kW, test.kH, test.dW, test.dH, test.padW, test.padH, "AVE")
      val prototxt = getPrototxt(test, "AVE")

      val restOfSentence = (s"with parameters ${test.batchSize}", test.channel, test.height,
        test.width, test.kW, test.kH, test.dW, test.dH, test.padW, test.padH)
        .productIterator.mkString(",")

      "A AveragePooling" should restOfSentence in {
        doTest(prototxt, model, test) // average pooling is AVE in caffe.
      }
    }
  }

  def diffBatchSizeOnSameModel(): Unit = {
    for (poolType <- List("MAX", "AVE")) {
      val initTest = TestCase(128, 128, 16, 16, 2, 2, 2, 2, 0, 0)
      val model = getModel(initTest, poolType)

      for (batchSize <- List(158, 295, 854)) {
        val test = TestCase(batchSize, 128, 16, 16, 2, 2, 2, 2, 0, 0)
        val prototxt = getPrototxt(test, poolType)

        val restOfSentence = (s"with parameters ${test.batchSize}", test.channel, test.height,
          test.width, test.kW, test.kH, test.dW, test.dH, test.padW, test.padH)
          .productIterator.mkString(",")

        s"A $poolType Pooling" should restOfSentence in {
          doTest(prototxt, model, test) // average pooling is AVE in caffe.
        }
      }
    }
  }

  def run(): Unit = {
    diffAttributes()
    diffBatchSizeOnSameModel()
  }

  run()

  case class TestCase(batchSize: Int, channel: Int, height: Int, width: Int,
                      kW: Int, kH: Int, dW: Int, dH: Int, padW: Int, padH: Int)
}
