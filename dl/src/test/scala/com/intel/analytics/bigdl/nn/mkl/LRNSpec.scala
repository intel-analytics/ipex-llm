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
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.Tensor

import scala.reflect.ClassTag

class LRNSpec extends FlatSpec with Matchers {
/*  "LRN output and gradient input" should "generate correct result" in {
    def test[T: ClassTag]()(implicit ev: TensorNumeric[T]) {
      val modelDnn = new LocalNormalizationAcrossChannels[T](5, 0.0001, 0.75)
      val modelBlas = new nn.LocalNormalizationAcrossChannels[T](5, 0.0001, 0.75)

      for (i <- 0 until Tools.getRandTimes()) {
        val input = Tensor[T](Array(32, 64, 112, 112)).fill(ev.fromType(0.1))

        modelDnn.forward(input)
        modelBlas.forward(input)

        Tools.printTensor(modelDnn.output, msg = "dnn output")
        Tools.printTensor(modelBlas.output, msg = "blas output")
        Tools.averageAll(modelDnn.output, "dnn output")
        Tools.averageAll(modelBlas.output, "blas output")

        val gradOutput = Tensor[T]().resizeAs(modelDnn.output).fill(ev.fromType(0.1))

        modelDnn.backward(input, gradOutput)
        modelBlas.backward(input, gradOutput)

        Tools.printTensor(modelDnn.gradInput, msg = "dnn gradinput")
        Tools.printTensor(modelBlas.gradInput, msg = "blas gradinput")
        Tools.averageAll(modelDnn.gradInput, "dnn gradient input")
        Tools.averageAll(modelBlas.gradInput, "blas gradient input")
        Tools.cumulativeError(modelDnn.output, modelBlas.output, "output") should be(0.0 +- 1e-6)
        Tools.cumulativeError(modelDnn.gradInput, modelBlas.gradInput, "gradient input") should be(
          0.0 +- 1e-6)
      }
    }

    test[Float]()
  }

  "LRN output and gradient input compared with caffe" should "is right" in {
    val modelDnn = new LocalNormalizationAcrossChannels[Float](5, 0.0001, 0.75)

    val input = Tools.getTensorFloat("input", Array(32, 64, 112, 112))
    modelDnn.forward(input)
    val output = Tools.getTensorFloat("output", modelDnn.output.size())

    Tools.printTensor(modelDnn.output, msg = "dnn output")
    Tools.printTensor(output, msg = "caffe output")
    Tools.averageAll(modelDnn.output, "dnn output")
    Tools.averageAll(output, "caffe output")

    val gradOutput = Tools.getTensorFloat("gradOutput", output.size())
    val gradInput = Tools.getTensorFloat("gradInput", input.size())

    modelDnn.backward(input, gradOutput)

    Tools.printTensor(modelDnn.gradInput, msg = "dnn gradinput")
    Tools.printTensor(gradInput, msg = "blas gradinput")
    Tools.averageAll(modelDnn.gradInput, "dnn gradient input")
    Tools.averageAll(gradInput, "blas gradient input")

    Tools.cumulativeError(modelDnn.output, output, "output") should be(0.0 +- 1e-6)
    Tools.cumulativeError(modelDnn.gradInput, gradInput, "gradient input") should be(0.0 +- 1e-6)
  }*/

  val testCases = List(
    // AlexNet
    TestCase(4, 96, 55, 55, 5, 0.0001, 0.75, 1.0),
    TestCase(4, 256, 27, 27, 5, 0.0001, 0.75, 1.0),

    // GoogleNet
    TestCase(8, 64, 56, 56, 5, 1.0E-4, 0.75, 1.0),
    TestCase(8, 192, 56, 56, 5, 1.0E-4, 0.75, 1.0)
  )

  import scala.sys.process._
  val cmd1 = "/home/wyz/workspace/caffe.intel/build/tools/test_lrn "
  for (test <- testCases) {
    "A SpatialCrossLRN" should s"with parameters " +
                                  s"${test.batchSize}, ${test.channel}, ${test.height}, ${test.width}" +
                                  ", " + s"${test.size}, ${test.alpha}, ${test.beta}, ${test.k}" in {
      val model = new SpatialCrossMapLRN[Float](test.size, test.alpha, test.beta, test.k)

      val cmd = (cmd1, test.batchSize, test.channel, test.height, test.width,
        test.size, test.alpha, test.beta, test.k).productIterator.mkString(" ")

      println(cmd)
      val ret = cmd.!!
      val pid = Tools.getPidFromString(ret)

      val input = Tools.getTensorFloat("input", Array(test.batchSize, test.channel,
                                                      test.width, test.height), pid)

      model.forward(input)

      val output = Tools.getTensorFloat("output", model.output.size(), pid)

      val gradOutput = Tools.getTensorFloat("gradOutput", output.size(), pid)
      val gradInput = Tools.getTensorFloat("gradInput", input.size(), pid)

      model.zeroGradParameters()
      model.backward(input, gradOutput)

      Tools.cumulativeError(model.output, output, "output") should be(0.0)
      Tools.cumulativeError(model.gradInput, gradInput, "gradient input") should be(0.0)
    }
  }

  case class TestCase(batchSize: Int , channel: Int , height: Int , width: Int , size: Int,
                      alpha: Double, beta: Double, k : Double)
}
