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
import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{FlatSpec, Matchers}

class LRNSpec extends FlatSpec with Matchers {
  val testCases = List(
    // AlexNet
    TestCase(4, 96, 55, 55, 5, 0.0001, 0.75, 1.0),
    TestCase(4, 256, 27, 27, 5, 0.0001, 0.75, 1.0),

    // GoogleNet
    TestCase(8, 64, 56, 56, 5, 1.0E-4, 0.75, 1.0),
    TestCase(8, 192, 56, 56, 5, 1.0E-4, 0.75, 1.0)
  )

  def getPrototxt(test: TestCase): String = {
    val prototxt =
      s"""
         |name: "LRN"
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
         |  name: "lrn"
         |  type: "LRN"
         |  bottom: "data"
         |  top: "lrn"
         |
         |  lrn_param {
         |    local_size: 5
         |    alpha: 0.0001
         |    beta: 0.75
         |    engine: MKL2017
         |  }
         |}
         """.stripMargin
    prototxt
  }

  def getModel(test: TestCase): SpatialCrossMapLRN[Float] = {
    new SpatialCrossMapLRN[Float](test.size, test.alpha, test.beta, test.k)
  }

  def testLayer(prototxt: String, model: SpatialCrossMapLRN[Float], test: TestCase): Unit = {
    val restOfSentence = (s"with parameters ${test.batchSize}", test.channel, test.height,
      test.width, test.size, test.alpha, test.beta, test.k).productIterator.mkString(",")

    "A SpatialCrossLRN" should restOfSentence in {
      val identity = Collect.run(prototxt)

      val input = loadTensor("Fwrd_data", Array(test.batchSize, test.channel,
        test.width, test.height), identity)

      for (i <- 0 until randTimes()) {
        model.forward(input)

        val tmp = Tensor().resizeAs(input)
        model.refs.input.backToUsr(tmp)

        val output = loadTensor("Fwrd_lrn", model.output.size(), identity)

        val gradOutput = loadTensor("Bwrd_lrn.loss", output.size(), identity)
        val gradInput = loadTensor("Bwrd_lrn", input.size(), identity)

        model.zeroGradParameters()
        model.backward(input, gradOutput)

        cumulativeError(model.output, output, "output") should be(0.0)
        cumulativeError(model.gradInput, gradInput, "gradient input") should be(0.0)
      }
    }
  }

  def diffAttributes(): Unit = {
    for (test <- testCases) {
      val prototxt = getPrototxt(test)
      val model = getModel(test)
      testLayer(prototxt, model, test)
    }
  }

  def diffBatchSizeOnSameModel(): Unit = {
    val initTest = TestCase(4, 96, 55, 55, 5, 0.0001, 0.75, 1.0)
    val model = getModel(initTest)
    for (batchSize <- List(8, 128, 234)) {
      val test = TestCase(batchSize, 96, 55, 55, 5, 0.0001, 0.75, 1.0)
      val prototxt = getPrototxt(test)

      testLayer(prototxt, model, test)
    }
  }

  def run(): Unit = {
    diffAttributes()
    diffBatchSizeOnSameModel()
  }

  run()

  case class TestCase(batchSize: Int, channel: Int, height: Int, width: Int, size: Int,
                      alpha: Double, beta: Double, k : Double)
}
