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
import com.intel.analytics.sparkdl.nn.Module
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.tensor.{Storage, Tensor}
import org.scalatest.{FlatSpec, Matchers}

import scala.reflect.ClassTag

class ConcatSpec extends FlatSpec with Matchers {
  "Concat only a SpatialConvolution layer" should "generate correct output and gradInput" in {
    val nInputPlane = 1
    val nOutputPlane = 1
    val kW = 2
    val kH = 2
    val dW = 1
    val dH = 1
    val padW = 0
    val padH = 0

    def test[T: ClassTag]()(implicit ev: TensorNumeric[T]): Unit = {
      val iH = 3
      val iW = 4
      val num = 3
      val oH = (iH + 2 * padH - kH) / dH + 1
      val oW = (iW + 2 * padW - kW) / dW + 1

      val kernel = Tensor[T](Array(kW, kH)).rand()
      val input = Tensor[T](Array(num, nInputPlane, iH, iW)).rand()
      val bias = Tensor[T](nInputPlane).rand()
      val gradOutput = Tensor[T](Array(3, nOutputPlane, oH, oW)).rand()

      val convDnn =
        new SpatialConvolution[T](nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
      convDnn.weight.copy(kernel)
      convDnn.bias.copy(bias)
      val concatDnn = new Concat[T](2)
      concatDnn.add(convDnn)

      val convBlas =
        new nn.SpatialConvolution[T](nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
      convBlas.weight.copy(kernel)
      convBlas.bias.copy(bias)
      val concatBlas = new nn.Concat[T](2)
      concatBlas.add(convBlas)

      val outputDnn = concatDnn.updateOutput(input)
      val outputBlas = concatBlas.updateOutput(input)

      val gradInputDnn = concatDnn.updateGradInput(input, gradOutput)
      val gradInputBlas = concatBlas.updateGradInput(input, gradOutput)

      outputDnn should be equals (outputBlas)
      gradInputDnn should be equals (gradInputBlas)
    }

    for (i <- 0 until 100) {
      test[Float]()
      test[Double]()
    }
  }

  "Concat with a Sequential" should "generate correct output" in {
    val nInputPlane = 1
    val nOutputPlane = 1
    val kW = 2
    val kH = 2
    val dW = 1
    val dH = 1
    val padW = 0
    val padH = 0

    def test[T: ClassTag]()(implicit ev: TensorNumeric[T]): Unit = {
      val iH = 3
      val iW = 4
      val num = 3
      val oH = (iH + 2 * padH - kH) / dH + 1
      val oW = (iW + 2 * padW - kW) / dW + 1

      val kernel = Tensor[T](Array(kW, kH)).rand()
      val input = Tensor[T](Array(num, nInputPlane, iH, iW)).rand()
      val bias = Tensor[T](nInputPlane).rand()
      val gradOutput = Tensor[T](Array(3, nOutputPlane, oH, oW)).rand()

      val convDnn =
        new SpatialConvolution[T](nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
      convDnn.weight.copy(kernel)
      convDnn.bias.copy(bias)
      val seqDnn = new nn.Sequential[T]
      seqDnn.add(convDnn)
      val concatDnn = new Concat[T](2)
      concatDnn.add(seqDnn)

      val convBlas =
        new nn.SpatialConvolution[T](nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
      convBlas.weight.copy(kernel)
      convBlas.bias.copy(bias)
      val seqBlas = new nn.Sequential[T]()
      seqBlas.add(convBlas)
      val concatBlas = new nn.Concat[T](2)
      concatBlas.add(seqBlas)

      val outputDnn = concatDnn.updateOutput(input)
      val outputBlas = concatBlas.updateOutput(input)

      val gradInputDnn = concatDnn.updateGradInput(input, gradOutput)
      val gradInputBlas = concatBlas.updateGradInput(input, gradOutput)

      outputDnn should be equals (outputBlas)
      gradInputDnn should be equals (gradInputBlas)
    }

    for (i <- 0 until 100) {
      test[Float]()
      test[Double]()
    }
  }

  "Concat with multi SpatialConvolution layers" should "generate correct gradient input" in {
    val nInputPlane = 1
    val nOutputPlane = 1
    val kW = 2
    val kH = 2
    val dW = 1
    val dH = 1
    val padW = 0
    val padH = 0

    def test[T: ClassTag]()(implicit ev: TensorNumeric[T]): Unit = {
      val iH = 3
      val iW = 4
      val num = 3
      val oH = (iH + 2 * padH - kH) / dH + 1
      val oW = (iW + 2 * padW - kW) / dW + 1
      val numConcats = scala.util.Random.nextInt(4 - 1) + 1
      println("numConcats = " + numConcats)

      val kernel = Tensor[T](Array(kW, kH)).rand()
      val input = Tensor[T](Array(num, nInputPlane, iH, iW)).rand()
      val bias = Tensor[T](nInputPlane).rand()
      val gradOutput =
        Tensor[T](Array(3, nOutputPlane, oH, oW)).rand().repeatTensor(Array(1, numConcats, 1, 1))

      println(input.size().mkString("\t"))
      println(gradOutput.size().mkString("\t"))

      val convDnn: Array[SpatialConvolution[T]] = new Array[SpatialConvolution[T]](numConcats)
      val convBlas: Array[nn.SpatialConvolution[T]] = new Array[nn.SpatialConvolution[T]](numConcats)

      val concatDnn = new Concat[T](2)
      val concatBlas = new nn.Concat[T](2)
      for (i <- 0 until numConcats) {
        convDnn(i) =
          new SpatialConvolution[T](nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
        convBlas(i) =
          new nn.SpatialConvolution[T](nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)

        convDnn(i).weight.copy(kernel)
        convDnn(i).bias.copy(bias)
        convBlas(i).weight.copy(kernel)
        convBlas(i).bias.copy(bias)

        concatDnn.add(convDnn(i))
        concatBlas.add(convBlas(i))
      }

      val outputDnn = concatDnn.updateOutput(input)
      val outputBlas = concatBlas.updateOutput(input)
      println(outputDnn)
      println(outputBlas)
      outputDnn should be equals (outputBlas)

      val gradInputDnn = concatDnn.updateGradInput(input, gradOutput)
      val gradInputBlas = concatBlas.updateGradInput(input, gradOutput)
      gradInputDnn should be equals (gradInputBlas)
    }

    for (i <- 0 until 100) {
      test[Float]()
    }
  }
}
