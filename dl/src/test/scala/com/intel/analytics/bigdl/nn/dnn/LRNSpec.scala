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

class LRNSpec extends FlatSpec with Matchers {
  val testCases = List(
    // AlexNet
    TestCase(4, 96, 55, 55, 5, 0.0001, 0.75, 1.0),
    TestCase(4, 256, 27, 27, 5, 0.0001, 0.75, 1.0),

    // GoogleNet
    TestCase(8, 64, 56, 56, 5, 1.0E-4, 0.75, 1.0),
    TestCase(8, 192, 56, 56, 5, 1.0E-4, 0.75, 1.0)
  )

  for (test <- testCases) {
    val restOfSentence = (s"with parameters ${test.batchSize}", test.channel, test.height,
      test.width, test.size, test.alpha, test.beta, test.k).productIterator.mkString(",")
    "A SpatialCrossLRN" should restOfSentence in {
      val prototxt =
        s"""
           |name: "LRN"
           |force_backward: true
           |layer {
           |  name: "data"
           |  type: "DummyData"
           |  top: "data"
           |  dummy_data_param {
           |    shape: { dim: 4 dim: 96 dim: 28 dim: 28}
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
      val identity = Collect.run(prototxt)

      val model = new SpatialCrossMapLRN[Float](test.size, test.alpha, test.beta, test.k)
      val input = Tools.loadTensor[Float]("Fwrd_data", Array(test.batchSize, test.channel,
                                                      test.width, test.height), identity)

      model.forward(input)

      val output = Tools.loadTensor[Float]("Fwrd_lrn", model.output.size(), identity)

      val gradOutput = Tools.loadTensor[Float]("Bwrd_lrn.loss", output.size(), identity)
      val gradInput = Tools.loadTensor[Float]("Bwrd_lrn", input.size(), identity)

      model.zeroGradParameters()
      model.backward(input, gradOutput)

      Tools.cumulativeError(model.output, output, "output") should be(0.0)
      Tools.cumulativeError(model.gradInput, gradInput, "gradient input") should be(0.0)
    }
  }

  case class TestCase(batchSize: Int, channel: Int, height: Int, width: Int, size: Int,
                      alpha: Double, beta: Double, k : Double)
}
