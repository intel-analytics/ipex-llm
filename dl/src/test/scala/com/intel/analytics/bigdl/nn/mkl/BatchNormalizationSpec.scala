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

package com.intel.analytics.bigdl.nn.mkl

import com.intel.analytics.bigdl.nn
import org.scalatest.{FlatSpec, Matchers}

class BatchNormalizationSpec extends FlatSpec with Matchers {
/*  "BatchNormalization output and gradInput compared with caffe" should "are the same" in {
    val modelDnn = new SpatialBatchNormalization[Float](64, 1e-3)
    val modelBlas = new nn.SpatialBatchNormalization[Float](64, 1e-3)

    val input = Tools.getTensorFloat("input", Array(32, 64, 112, 112))
    val weights = Tools.getTensorFloat("weights", Array(64))
    val bias = Tools.getTensorFloat("bias", Array(64))

    modelDnn.weight.set(weights)
    modelDnn.bias.set(bias)
    modelDnn.gradWeight.set(weights)
    modelDnn.gradBias.set(bias)
    modelBlas.weight.set(weights)
    modelBlas.bias.set(bias)

    modelDnn.forward(input)
    modelBlas.forward(input)

    val output = Tools.getTensorFloat("output", modelDnn.output.size())

    Tools.printTensor(modelDnn.output, msg = "dnn output")
    Tools.printTensor(output, msg = "caffe output")
    Tools.averageAll(modelDnn.output, "dnn output")
    Tools.averageAll(output, "caffe output")

    val gradOutput = Tools.getTensorFloat("gradOutput", output.size())
    val gradInput = Tools.getTensorFloat("gradInput", input.size())

    modelDnn.backward(input, gradOutput)
    modelBlas.backward(input, gradOutput)

    Tools.printTensor(modelDnn.gradInput, msg = "dnn gradinput")
    Tools.printTensor(gradInput, msg = "blas gradinput")
    Tools.averageAll(modelDnn.gradInput, "dnn gradient input")
    Tools.averageAll(gradInput, "blas gradient input")

    Tools.cumulativeError(modelDnn.output, output, "output") should be(0.0 +- 1e-6)
    Tools.cumulativeError(modelDnn.gradInput, gradInput, "gradient input") should be(0.0 +- 1e-6)

    val gradWeight = Tools.getTensorFloat("gradWeight", weights.size())
    val gradBias = Tools.getTensorFloat("gradBias", bias.size())

    Tools.averageAll(weights, "weights average")
    Tools.averageAll(bias, "bias average")
    Tools.cumulativeError(modelDnn.gradWeight, gradWeight, "weights") should be(0.0)
    Tools.cumulativeError(modelDnn.gradBias, gradBias, "bias") should be(0.0)

    Tools.cumulativeError(modelDnn.output, modelBlas.output, "output")
    Tools.cumulativeError(modelDnn.gradInput, modelBlas.gradInput, "gradient input")
  }
  "BatchNormalization 2-D output and gradInput compared with caffe" should "are the same" in {
    def test() {
      val modelDnn = new BatchNormalization[Float](64, 1e-3)
      val modelBlas = new nn.SpatialBatchNormalization[Float](64, 1e-3)

      val input = Tools.getTensorFloat("input", Array(128, 64, 32, 32))
      val weights = Tools.getTensorFloat("weights", Array(64))
      val bias = Tools.getTensorFloat("bias", Array(64))

      modelDnn.weight.set(weights)
      modelDnn.bias.set(bias)
      modelBlas.weight.set(weights)
      modelBlas.bias.set(bias)

      modelDnn.forward(input)
      modelBlas.forward(input)

      val output = Tools.getTensorFloat("output", modelDnn.output.size())

      val gradOutput = Tools.getTensorFloat("gradOutput", output.size())
      val gradInput = Tools.getTensorFloat("gradInput", input.size())

      modelDnn.backward(input, gradOutput)
      modelBlas.backward(input, gradOutput)

      Tools.cumulativeError(modelDnn.output, output,
                            "compare caffe output") should be(0.0)
      Tools.cumulativeError(modelDnn.gradInput, gradInput,
                            "compare caffe gradient input") should be(0.0)

      val gradWeight = Tools.getTensorFloat("gradWeight", weights.size())
      val gradBias = Tools.getTensorFloat("gradBias", bias.size())

      Tools.cumulativeError(modelDnn.gradWeight, gradWeight,
                            "compare caffe gradient weights") should be(0.0)
      Tools.cumulativeError(modelDnn.gradBias, gradBias,
                            "compare caffe gradient bias") should be(0.0)

      Tools.cumulativeError(modelDnn.gradWeight, weights, "MUST NOT BE SAME")

      Tools.cumulativeError(modelDnn.output, modelBlas.output,
                            "compare blas output") should be (0.0 +- 1e-4)
      Tools.cumulativeError(modelDnn.gradInput, modelBlas.gradInput,
                            "compare blas gradient input") should be (0.0 +- 1e-4)
      Tools.cumulativeError(modelDnn.gradWeight, modelBlas.gradWeight,
                            "compare blas gradient weights") should be(0.0 +- 1e-4)
      Tools.cumulativeError(modelDnn.gradBias, modelBlas.gradBias,
                            "compare blas gradient bias") should be(0.0 +- 1e-4)
    }
    test()
  }*/

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
    TestCase(128, 96, 28, 28, 0.001)
  )

  import scala.sys.process._
  val cmd1 = "/home/wyz/workspace/caffe.intel/build/tools/test_batch_norm"
  for (test <- testCases) {
    "A BatchNormalization" should s"with parameters " +
                                  s"${test.batchSize}, ${test.channel}, ${test.height}," +
                                  ", " + s"${test.width}, ${test.eps}" in {
      val model = new BatchNormalization[Float](test.channel, test.eps)

      val cmd = (cmd1, test.batchSize, test.channel, test.height, test.width, test.eps)
        .productIterator.mkString(" ")

      println(cmd)
      val ret = cmd.!!
      val pid = Tools.getPidFromString(ret)

      val input = Tools.getTensorFloat("input", Array(test.batchSize, test.channel,
                                                      test.width, test.height), pid)
      val weights = Tools.getTensorFloat("weights", model.weight.size(), pid)
      val bias = Tools.getTensorFloat("bias", Array(test.channel), pid)

      model.weight.set(weights)
      model.bias.set(bias)

      model.forward(input)

      val output = Tools.getTensorFloat("output", model.output.size(), pid)

      val gradOutput = Tools.getTensorFloat("gradOutput", output.size(), pid)
      val gradInput = Tools.getTensorFloat("gradInput", input.size(), pid)

      model.zeroGradParameters()
      model.backward(input, gradOutput)

      val gradWeight = Tools.getTensorFloat("gradWeight", weights.size(), pid)
      val gradBias = Tools.getTensorFloat("gradBias", bias.size(), pid)

      Tools.cumulativeError(model.output, output, "output") should be(0.0)
      Tools.cumulativeError(model.gradInput, gradInput, "gradient input") should be(0.0)
      Tools.cumulativeError(model.gradWeight, gradWeight, "gradWeight") should be(0.0)
      Tools.cumulativeError(model.gradBias, gradBias, "gradBias") should be(0.0)
    }
  }

  case class TestCase(batchSize: Int , channel: Int , height: Int , width: Int , eps: Double)
}
