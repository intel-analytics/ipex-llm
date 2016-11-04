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

package com.intel.analytics.sparkdl.nn.mkl

import com.intel.analytics.sparkdl.nn
import com.intel.analytics.sparkdl.nn._
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.tensor.{Storage, Tensor}
import org.scalatest.{FlatSpec, Matchers}
import scala.sys.process._

import scala.reflect.ClassTag
import scala.tools.nsc.Phases.Model
class PoolingSpec extends FlatSpec with Matchers {
/*  "SpatialMaxPooling ceil mode" should "generate correct output and gradient input" in {
    def test[T: ClassTag]()(implicit ev: TensorNumeric[T]): Unit = {
      val maxPoolDnn = new SpatialMaxPooling[T](3, 3, 2, 2).ceil()
      val maxPoolBlas = new nn.SpatialMaxPooling[T](3, 3, 2, 2).ceil()

      for (i <- 0 until 5) {
        val input = Tensor[T](32, 64, 112, 112).rand()

        val outputDnn = maxPoolDnn.forward(input)
        val outputBlas = maxPoolBlas.forward(input)

        Tools.averageError(outputDnn, outputBlas, "output") should be(0.0 +- 1e-6)

        val gradOutput = Tensor[T]().resizeAs(outputDnn).rand()

        val gradInputDnn = maxPoolDnn.backward(input, gradOutput)
        val gradInputBlas = maxPoolBlas.backward(input, gradOutput)

        Tools.cumulativeError(gradInputDnn, gradInputBlas, "gradOutput")
        Tools.averageError(gradInputDnn, gradInputBlas, "gradOutput") should be(0.0 +- 1e-6)
      }
    }

    for (i <- 0 until Tools.getRandTimes()) {
      test[Float]()
    }
  }

  "SpatialAvergePooling ceil mode" should "generate correct output and gradient input" in {
    def test[T: ClassTag]()(implicit ev: TensorNumeric[T]): Unit = {
      val maxPoolDnn = new SpatialAveragePooling[T](5, 5, 3, 3).ceil()
      val maxPoolBlas = new nn.SpatialAveragePooling[T](5, 5, 3, 3).ceil()

      for (i <- 0 until 5) {
        val input = Tensor[T](8, 64, 112, 112).rand()

        val outputDnn = maxPoolDnn.forward(input)
        val outputBlas = maxPoolBlas.forward(input)

        Tools.averageError(outputDnn, outputBlas, "output") should be(0.0 +- 1e-6)

        val gradOutput = Tensor[T]().resizeAs(outputDnn).rand()

        val gradInputDnn = maxPoolDnn.backward(input, gradOutput)
        val gradInputBlas = maxPoolBlas.backward(input, gradOutput)

        Tools.cumulativeError(gradInputDnn, gradInputBlas, "gradOutput")
        Tools.averageError(gradInputDnn, gradInputBlas, "gradOutput") should be(0.0 +- 1e-6)
      }
    }

    for (i <- 0 until Tools.getRandTimes()) {
      test[Float]()
      test[Double]()
    }
  }
  "SpatialAvergePooling ceil mode 7 7 1 1" should "generate correct output and gradient input" in {
    def test[T: ClassTag]()(implicit ev: TensorNumeric[T]): Unit = {
      val maxPoolDnn = new SpatialAveragePooling[T](7, 7, 1, 1).ceil()
      val maxPoolBlas = new nn.SpatialAveragePooling[T](7, 7, 1, 1).ceil()

      for (i <- 0 until 5) {
        val input = Tensor[T](8, 1024, 7, 7).rand()

        val outputDnn = maxPoolDnn.forward(input)
        val outputBlas = maxPoolBlas.forward(input)

        Tools.averageError(outputDnn, outputBlas, "output") should be(0.0 +- 1e-6)

        val gradOutput = Tensor[T]().resizeAs(outputDnn).rand()

        val gradInputDnn = maxPoolDnn.backward(input, gradOutput)
        val gradInputBlas = maxPoolBlas.backward(input, gradOutput)

        Tools.cumulativeError(gradInputDnn, gradInputBlas, "gradInput")
        Tools.averageError(gradInputDnn, gradInputBlas, "gradOutput") should be(0.0 +- 1e-6)
      }
    }

    for (i <- 0 until Tools.getRandTimes()) {
      test[Float]()
      test[Double]()
    }
  }*/

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

  def getModel(kW: Int, kH: Int, dW: Int, dH: Int,
               padW: Int, padH: Int, ver : String) : SpatialPooling[Float] = {
    ver match {
      case "MAX" =>
        new SpatialMaxPooling[Float](kW, kH, dW, dH, padW, padH).ceil()
      case "AVG" =>
        new SpatialAveragePooling[Float](kW, kH, dW, dH, padW, padH).ceil()
    }
  }

  def doTest(test: TestCase, cmd1: String, model : Module[Float]) : Unit = {
    val cmd = (cmd1, test.batchSize, test.channel, test.height, test.width,
      test.kW, test.kH, test.dW, test.dH, test.padW, test.padH)
      .productIterator.mkString(" ")

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

  for (test <- testCases) {
    "A MaxPooling" should s"with parameters " +
                                  s"${test.batchSize}, ${test.channel}, ${test.height}" +
                                  ", " + s"${test.width}, ${test.kW}, ${test.kH}" +
                          " " + s"${test.dW}, ${test.dH}, ${test.padW}, ${test.padH}" in {
      val cmd1 = "/home/wyz/workspace/caffe.intel/build/tools/test_max_pooling"
      doTest(test, cmd1, getModel(test.kW, test.kH, test.dW, test.dH, test.padW, test.padH, "MAX"))
    }
  }

  for (test <- testCases) {
    "A AveragePooling" should s"with parameters " +
                          s"${test.batchSize}, ${test.channel}, ${test.height}" +
                          ", " + s"${test.width}, ${test.kW}, ${test.kH}" +
                          " " + s"${test.dW}, ${test.dH}, ${test.padW}, ${test.padH}" in {
      val cmd1 = "/home/wyz/workspace/caffe.intel/build/tools/test_avg_pooling"
      doTest(test, cmd1, getModel(test.kW, test.kH, test.dW, test.dH, test.padW, test.padH, "AVG"))
    }
  }

  case class TestCase(batchSize: Int , channel: Int , height: Int , width: Int,
                      kW: Int, kH: Int, dW: Int, dH:Int, padW: Int, padH: Int)
}
