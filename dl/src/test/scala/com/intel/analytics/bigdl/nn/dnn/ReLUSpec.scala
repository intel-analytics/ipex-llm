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

class ReLUSpec extends FlatSpec with Matchers {
  val testCases = List(
    TestCase(4, 96, 55, 55, ip = false),
    TestCase(4, 256, 27, 27, ip = false),
    TestCase(4, 384, 13, 13, ip = false),
    TestCase(4, 256, 13, 13, ip = false),
//    TestCase(4, 4096, false),

    TestCase(4, 96, 55, 55, ip = true),
    TestCase(4, 256, 27, 27, ip = true),
    TestCase(4, 384, 13, 13, ip = true),
    TestCase(4, 256, 13, 13, ip = true)
//    TestCase(4, 4096, true)
  )

  for (test <- testCases) {
    val restOfSentence = (s"with parameters ${test.batchSize}", test.channel, test.height,
      test.width, test.ip).productIterator.mkString(",")

    "A SpatialCrossReLU" should restOfSentence in {
      val prototxt =
        s"""
           |name: "ReLU"
           |force_backward: true
           |layer {
           |  name: "data"
           |  type: "DummyData"
           |  top: "data"
           |  dummy_data_param {
           |    shape: {
           |      dim: ${test.batchSize}
           |      dim: ${test.channel}
           |      dim: ${test.height},
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
           |  name: "relu"
           |  type: "ReLU"
           |  bottom: "data"
           |  top: "relu"
           |
           |  relu_param {
           |    engine: MKL2017
           |  }
           |}
         """.stripMargin
      val identity = Collect.run(prototxt)

      val model = new ReLU[Float](test.ip)
      val input = Tools.loadTensor[Float]("Fwrd_data", Array(test.batchSize, test.channel,
        test.width, test.height), identity)

      model.forward(input)

      val output = Tools.loadTensor[Float]("Fwrd_relu", model.output.size(), identity)
      val gradOutput = Tools.loadTensor[Float]("Bwrd_relu.loss", output.size(), identity)
      val gradInput = Tools.loadTensor[Float]("Bwrd_relu", input.size(), identity)

      model.zeroGradParameters()
      model.backward(input, gradOutput)

      Tools.cumulativeError(model.output, output, "output") should be(0.0)
      Tools.cumulativeError(model.gradInput, gradInput, "gradient input") should be(0.0)
    }
  }

  case class TestCase(batchSize: Int, channel: Int, height: Int, width: Int, ip: Boolean)
}

