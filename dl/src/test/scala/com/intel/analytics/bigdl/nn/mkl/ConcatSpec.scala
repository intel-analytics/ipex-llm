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
import com.intel.analytics.bigdl.nn.{Constant, Default, Module, Xavier}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.utils.RandomGenerator._

import scala.reflect.ClassTag

class ConcatSpec extends FlatSpec with Matchers {
  def error2Tensor[T: ClassTag](tensor1: Tensor[T], tensor2: Tensor[T])(
      implicit ev: TensorNumeric[T]): Double = {
    require(tensor1.nElement() == tensor2.nElement())
    var tmp = 0.0
    for (i <- 0 until tensor1.nElement()) {
      tmp += math.abs(
        ev.toType[Double](tensor1.storage().array()(i)) -
          ev.toType[Double](tensor2.storage().array()(i)))
    }
    println("ERROR: " + tmp)
    tmp
  }

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

      val gradInputDnn = concatDnn.backward(input, gradOutput)
      val gradInputBlas = concatBlas.backward(input, gradOutput)

      outputDnn should be equals (outputBlas)
      gradInputDnn should be equals (gradInputBlas)

      error2Tensor[T](outputDnn, outputBlas) should be(0.0 +- 2 * 1e-6)
      error2Tensor[T](gradInputDnn, gradInputBlas) should be(0.0 +- 2 * 1e-6)
    }

    for (i <- 0 until Tools.getRandTimes()) {
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
      val seqDnn = new nn.Sequential[Tensor[T], Tensor[T], T]
      seqDnn.add(convDnn)
      val concatDnn = new Concat[T](2)
      concatDnn.add(seqDnn)

      val convBlas =
        new nn.SpatialConvolution[T](nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
      convBlas.weight.copy(kernel)
      convBlas.bias.copy(bias)
      val seqBlas = new nn.Sequential[Tensor[T], Tensor[T], T]()
      seqBlas.add(convBlas)
      val concatBlas = new nn.Concat[T](2)
      concatBlas.add(seqBlas)

      val outputDnn = concatDnn.updateOutput(input)
      val outputBlas = concatBlas.updateOutput(input)

      val gradInputDnn = concatDnn.backward(input, gradOutput)
      val gradInputBlas = concatBlas.backward(input, gradOutput)

      outputDnn should be equals (outputBlas)
      gradInputDnn should be equals (gradInputBlas)

      error2Tensor[T](outputDnn, outputBlas) should be(0.0 +- 2 * 1e-6)
      error2Tensor[T](gradInputDnn, gradInputBlas) should be(0.0 +- 2 * 1e-6)
    }

    for (i <- 0 until Tools.getRandTimes()) {
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
      val convBlas: Array[nn.SpatialConvolution[T]] =
        new Array[nn.SpatialConvolution[T]](numConcats)

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

      val gradInputDnn = concatDnn.backward(input, gradOutput)
      val gradInputBlas = concatBlas.backward(input, gradOutput)
      println(gradInputDnn)
      println(gradInputBlas)
      gradInputDnn should be equals (gradInputBlas)

      // TODO 1e-5 is allowable ?
      error2Tensor[T](outputDnn, outputBlas) should be(0.0 +- 1e-5)
      error2Tensor[T](gradInputDnn, gradInputBlas) should be(0.0 +- 1e-5)
    }

    for (i <- 0 until Tools.getRandTimes()) {
      test[Float]()
      test[Double]()
    }
  }

  "Concat with multi sequential" should "generate correct output and gradient input" in {
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
      val convBlas: Array[nn.SpatialConvolution[T]] =
        new Array[nn.SpatialConvolution[T]](numConcats)

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

        val seqDnn = new nn.Sequential[Tensor[T], Tensor[T], T]()
        val seqBlas = new nn.Sequential[Tensor[T], Tensor[T], T]()

        seqDnn.add(convDnn(i))
        seqBlas.add(convBlas(i))

        concatDnn.add(seqDnn)
        concatBlas.add(seqBlas)
      }

      val outputDnn = concatDnn.updateOutput(input)
      val outputBlas = concatBlas.updateOutput(input)
      println(outputDnn)
      println(outputBlas)
      outputDnn should be equals (outputBlas)

      val gradInputDnn = concatDnn.backward(input, gradOutput)
      val gradInputBlas = concatBlas.backward(input, gradOutput)
      println(gradInputDnn)
      println(gradInputBlas)
      gradInputDnn should be equals (gradInputBlas)
      // TODO 1e-5 is allowable ?
      error2Tensor[T](outputDnn, outputBlas) should be(0.0 +- 1e-5)
      error2Tensor[T](gradInputDnn, gradInputBlas) should be(0.0 +- 1e-5)
    }

    for (i <- 0 until Tools.getRandTimes()) {
      test[Float]()
      test[Double]()
    }
  }

  "Concat with GoogLeNet inception contains all nn layers" should "generate correct results" in {
    def model[T: ClassTag]()(implicit ev: TensorNumeric[T]): Module[Tensor[T], Tensor[T], T] = {
      val concat = new Concat[T](2)

      val conv1 = new nn.Sequential[Tensor[T], Tensor[T], T]()
      val conv3 = new nn.Sequential[Tensor[T], Tensor[T], T]()
      val conv5 = new nn.Sequential[Tensor[T], Tensor[T], T]()
      val pool = new nn.Sequential[Tensor[T], Tensor[T], T]()

      conv1.add(new nn.SpatialConvolution[T](192, 64, 1, 1, 1, 1, 0, 0).setInitMethod(Xavier))
      conv1.add(new nn.ReLU[T](true))

      conv3.add(new nn.SpatialConvolution[T](192, 96, 1, 1, 1, 1, 0, 0).setInitMethod(Xavier))
      conv3.add(new nn.ReLU[T](true))
      conv3.add(new nn.SpatialConvolution[T](96, 128, 3, 3, 1, 1, 1, 1).setInitMethod(Xavier))
      conv3.add(new nn.ReLU[T](true))

      conv5.add(new nn.SpatialConvolution[T](192, 16, 1, 1, 1, 1, 0, 0).setInitMethod(Xavier))
      conv5.add(new nn.ReLU[T](true))
      conv5.add(new nn.SpatialConvolution[T](16, 32, 5, 5, 1, 1, 2, 2).setInitMethod(Xavier))
      conv5.add(new nn.ReLU[T](true))

      pool.add(new nn.SpatialMaxPooling[T](3, 3, 1, 1, 1, 1).ceil())
      pool.add(new nn.SpatialConvolution[T](192, 32, 1, 1, 1, 1, 0, 0).setInitMethod(Xavier))
      pool.add(new nn.ReLU[T](true))

      concat.add(conv1)
      concat.add(conv3)
      concat.add(conv5)
      concat.add(pool)
      concat
    }

    def test[T: ClassTag]()(implicit ev: TensorNumeric[T]): Unit = {
      val dnn1 = model[T]()
      val dnn2 = model[T]()

      val dnn1Para = dnn1.parameters()
      val dnn2Para = dnn2.parameters()
      for (i <- 0 until dnn1Para._1.length) {
        dnn1Para._1(i).copy(dnn2Para._1(i))
      }

      val input = Tensor[T](Array(32, 192, 28, 28)).rand()
      val gradOutput = Tensor[T](Array(32, 256, 28, 28)).rand()

      val output1 = dnn1.updateOutput(input)
      val output2 = dnn2.updateOutput(input)
      output1 should be equals (output2)

      output1.nElement() should be(output2.nElement())

      val gradInputDnn1 = dnn1.backward(input, gradOutput)
      val gradInputDnn2 = dnn2.backward(input, gradOutput)
      gradInputDnn1 should be equals (gradInputDnn2)

      Tools.averageError[T](output1, output2, "output") should be(0.0 +- 1e-6)
      Tools.averageError[T](gradInputDnn1, gradInputDnn2, "gradinput") should be(0.0 +- 1e-6)
    }

    for (i <- 0 until Tools.getRandTimes()) {
      test[Float]()
      test[Double]()
    }
  }

  "Concat with GoogLeNet inception contains all mkl layers" should "generate correct results" in {
    def model[T: ClassTag]()(implicit ev: TensorNumeric[T]): Module[Tensor[T], Tensor[T], T] = {
      val concat = new Concat[T](2)

      val conv1 = new nn.Sequential[Tensor[T], Tensor[T], T]()
      val conv3 = new nn.Sequential[Tensor[T], Tensor[T], T]()
      val conv5 = new nn.Sequential[Tensor[T], Tensor[T], T]()
      val pool  = new nn.Sequential[Tensor[T], Tensor[T], T]()

      conv1.add(new SpatialConvolution[T](192, 64, 1, 1, 1, 1, 0, 0).setInitMethod(Xavier))
      conv1.add(new ReLU[T](true))

      conv3.add(new SpatialConvolution[T](192, 96, 1, 1, 1, 1, 0, 0).setInitMethod(Xavier))
      conv3.add(new ReLU[T](true))
      conv3.add(new SpatialConvolution[T](96, 128, 3, 3, 1, 1, 1, 1).setInitMethod(Xavier))
      conv3.add(new ReLU[T](true))

      conv5.add(new SpatialConvolution[T](192, 16, 1, 1, 1, 1, 0, 0).setInitMethod(Xavier))
      conv5.add(new ReLU[T](true))
      conv5.add(new SpatialConvolution[T](16, 32, 5, 5, 1, 1, 2, 2).setInitMethod(Xavier))
      conv5.add(new ReLU[T](true))

      pool.add(new SpatialMaxPooling[T](3, 3, 1, 1, 1, 1).ceil())
      pool.add(new SpatialConvolution[T](192, 32, 1, 1, 1, 1, 0, 0).setInitMethod(Xavier))
      pool.add(new ReLU[T](true))

      concat.add(conv1)
      concat.add(conv3)
      concat.add(conv5)
      concat.add(pool)
      concat
    }

    def test[T: ClassTag]()(implicit ev: TensorNumeric[T]): Unit = {
      val dnn1 = model[T]()
      val dnn2 = model[T]()

      val dnn1Para = dnn1.parameters()
      val dnn2Para = dnn2.parameters()
      for (i <- 0 until dnn1Para._1.length) {
        dnn1Para._1(i).copy(dnn2Para._1(i))
      }

      val input = Tensor[T](Array(32, 192, 28, 28)).rand()
      val gradOutput = Tensor[T](Array(32, 256, 28, 28)).rand()

      val output1 = dnn1.updateOutput(input)
      val output2 = dnn2.updateOutput(input)
      output1 should be equals (output2)

      output1.nElement() should be(output2.nElement())

      val gradInputDnn1 = dnn1.backward(input, gradOutput)
      val gradInputDnn2 = dnn2.backward(input, gradOutput)
      gradInputDnn1 should be equals (gradInputDnn2)

      Tools.averageError[T](output1, output2, "output") should be(0.0 +- 1e-6)
      Tools.averageError[T](gradInputDnn1, gradInputDnn2, "gradinput") should be(0.0 +- 1e-6)
    }

    for (i <- 0 until Tools.getRandTimes()) {
      test[Float]()
      test[Double]()
    }
  }

  "Concat contains two version of layers" should "generate correct results" in {
    def model[T: ClassTag](backend: String)(implicit ev: TensorNumeric[T]): Module[Tensor[T], Tensor[T], T] = {
      backend match {
        case "dnn" =>
          val concat = new Concat[T](2)

          val conv1 = new nn.Sequential[Tensor[T], Tensor[T], T]()
          val conv3 = new nn.Sequential[Tensor[T], Tensor[T], T]()
          val conv5 = new nn.Sequential[Tensor[T], Tensor[T], T]()
          val pool = new nn.Sequential[Tensor[T], Tensor[T], T]()

          conv1.add(new SpatialConvolution[T](192, 64, 1, 1, 1, 1, 0, 0).setInitMethod(Xavier))
          conv1.add(new ReLU[T](true))

          conv3.add(new SpatialConvolution[T](192, 96, 1, 1, 1, 1, 0, 0).setInitMethod(Xavier))
          conv3.add(new ReLU[T](true))
          conv3.add(new SpatialConvolution[T](96, 128, 3, 3, 1, 1, 1, 1).setInitMethod(Xavier))
          conv3.add(new ReLU[T](true))

          conv5.add(new SpatialConvolution[T](192, 16, 1, 1, 1, 1, 0, 0).setInitMethod(Xavier))
          conv5.add(new ReLU[T](true))
          conv5.add(new SpatialConvolution[T](16, 32, 5, 5, 1, 1, 2, 2).setInitMethod(Xavier))
          conv5.add(new ReLU[T](true))

          pool.add(new SpatialMaxPooling[T](3, 3, 1, 1, 1, 1).ceil())
          pool.add(new SpatialConvolution[T](192, 32, 1, 1, 1, 1, 0, 0).setInitMethod(Xavier))
          pool.add(new ReLU[T](true))

          concat.add(conv1)
          concat.add(conv3)
          concat.add(conv5)
          concat.add(pool)
          concat

        case "blas" =>
          val concat = new nn.Concat[T](2)

          val conv1 = new nn.Sequential[Tensor[T], Tensor[T], T]()
          val conv3 = new nn.Sequential[Tensor[T], Tensor[T], T]()
          val conv5 = new nn.Sequential[Tensor[T], Tensor[T], T]()
          val pool = new nn.Sequential[Tensor[T], Tensor[T], T]()

          conv1.add(new nn.SpatialConvolution[T](192, 64, 1, 1, 1, 1, 0, 0).setInitMethod(Xavier))
          conv1.add(new nn.ReLU[T](true))

          conv3.add(new nn.SpatialConvolution[T](192, 96, 1, 1, 1, 1, 0, 0).setInitMethod(Xavier))
          conv3.add(new nn.ReLU[T](true))
          conv3.add(new nn.SpatialConvolution[T](96, 128, 3, 3, 1, 1, 1, 1).setInitMethod(Xavier))
          conv3.add(new nn.ReLU[T](true))

          conv5.add(new nn.SpatialConvolution[T](192, 16, 1, 1, 1, 1, 0, 0).setInitMethod(Xavier))
          conv5.add(new nn.ReLU[T](true))
          conv5.add(new nn.SpatialConvolution[T](16, 32, 5, 5, 1, 1, 2, 2).setInitMethod(Xavier))
          conv5.add(new nn.ReLU[T](true))

          pool.add(new nn.SpatialMaxPooling[T](3, 3, 1, 1, 1, 1).ceil())
          pool.add(new nn.SpatialConvolution[T](192, 32, 1, 1, 1, 1, 0, 0).setInitMethod(Xavier))
          pool.add(new nn.ReLU[T](true))

          concat.add(conv1)
          concat.add(conv3)
          concat.add(conv5)
          concat.add(pool)
          concat
      }
    }

    def test[T: ClassTag]()(implicit ev: TensorNumeric[T]): Unit = {
      val dnn = model[T]("dnn")
      val blas = model[T]("blas")

      val dnnPara = dnn.parameters()
      val blasPara = blas.parameters()
      for (i <- 0 until dnnPara._1.length) {
        dnnPara._1(i).copy(blasPara._1(i))
      }

      val input = Tensor[T](Array(32, 192, 28, 28)).rand()
      val gradOutput = Tensor[T](Array(32, 256, 28, 28)).rand()

      val outputDnn = dnn.updateOutput(input)
      val outputBlas = blas.updateOutput(input)
      outputDnn should be equals (outputBlas)

      outputDnn.nElement() should be(outputBlas.nElement())

      val gradInputDnn = dnn.backward(input, gradOutput)
      val gradInputBlas = blas.backward(input, gradOutput)
      gradInputDnn should be equals (gradInputBlas)

      Tools.averageError[T](outputDnn, outputBlas, "output") should be(0.0 +- 1e-5)
      Tools.averageError[T](gradInputDnn, gradInputBlas, "gradinput") should be(0.0 +- 1e-5)
    }

    for (i <- 0 until Tools.getRandTimes()) {
      test[Float]()
      test[Double]()
    }
  }

  "Concat with GoogLeNet inception contains mix backend" should "generate correct result" in {
    def model[T: ClassTag](backend: String)
                          (implicit ev: TensorNumeric[T]): Module[Tensor[T], Tensor[T], T] = {
      backend match {
        case "mix" =>
          val concat = new Concat[T](2)

          val conv1 = new nn.Sequential[Tensor[T], Tensor[T], T]()
          val conv3 = new nn.Sequential[Tensor[T], Tensor[T], T]()
          val conv5 = new nn.Sequential[Tensor[T], Tensor[T], T]()
          val pool = new nn.Sequential[Tensor[T], Tensor[T], T]()

          val randNum = scala.util.Random

          def randModule(m1: () => Module[Tensor[T], Tensor[T], T],
                         m2: () => Module[Tensor[T], Tensor[T], T]):
          Module[Tensor[T], Tensor[T], T] = {
            if (randNum.nextInt(2) != 0) {
              m1()
            } else {
              m2()
            }
          }

          conv1.add(
            randModule(
              () => new SpatialConvolution[T](192, 64, 1, 1, 1, 1, 0, 0).setInitMethod(Xavier),
              () => new nn.SpatialConvolution[T](192, 64, 1, 1, 1, 1, 0, 0).setInitMethod(Xavier))
          )
          conv1.add(
            randModule(() => new ReLU[T](true), () => new nn.ReLU[T](true))
          )

          conv3.add(
            randModule(
              () => new SpatialConvolution[T](192, 96, 1, 1, 1, 1, 0, 0).setInitMethod(Xavier),
              () => new nn.SpatialConvolution[T](192, 96, 1, 1, 1, 1, 0, 0).setInitMethod(Xavier))
          )
          conv3.add(
            randModule(() => new ReLU[T](true), () => new nn.ReLU[T](true))
          )
          conv3.add(
            randModule(
              () => new SpatialConvolution[T](96, 128, 3, 3, 1, 1, 1, 1).setInitMethod(Xavier),
              () => new nn.SpatialConvolution[T](96, 128, 3, 3, 1, 1, 1, 1).setInitMethod(Xavier))
          )
          conv3.add(
            randModule(() => new ReLU[T](true), () => new nn.ReLU[T](true))
          )

          conv5.add(
            randModule(
              () => new SpatialConvolution[T](192, 16, 1, 1, 1, 1, 0, 0).setInitMethod(Xavier),
              () => new nn.SpatialConvolution[T](192, 16, 1, 1, 1, 1, 0, 0).setInitMethod(Xavier))
          )
          conv5.add(randModule(() => new ReLU[T](true), () => new nn.ReLU[T](true)))
          conv5.add(
            randModule(
              () => new SpatialConvolution[T](16, 32, 5, 5, 1, 1, 2, 2).setInitMethod(Xavier),
              () => new nn.SpatialConvolution[T](16, 32, 5, 5, 1, 1, 2, 2).setInitMethod(Xavier))
          )
          conv5.add(randModule(() => new ReLU[T](true), () => new nn.ReLU[T](true)))

          pool.add(
            randModule(() => new SpatialMaxPooling[T](3, 3, 1, 1, 1, 1).ceil(),
                       () => new nn.SpatialMaxPooling[T](3, 3, 1, 1, 1, 1).ceil())
          )
          pool.add(
            randModule(
              () => new SpatialConvolution[T](192, 32, 1, 1, 1, 1, 0, 0).setInitMethod(Xavier),
              () => new nn.SpatialConvolution[T](192, 32, 1, 1, 1, 1, 0, 0).setInitMethod(Xavier)
            )
          )
          pool.add(
            randModule(() => new ReLU[T](true), () => new nn.ReLU[T](true))
          )

          concat.add(conv1)
          concat.add(conv3)
          concat.add(conv5)
          concat.add(pool)
          concat

        case "blas" =>
          val concat = new nn.Concat[T](2)

          val conv1 = new nn.Sequential[Tensor[T], Tensor[T], T]()
          val conv3 = new nn.Sequential[Tensor[T], Tensor[T], T]()
          val conv5 = new nn.Sequential[Tensor[T], Tensor[T], T]()
          val pool = new nn.Sequential[Tensor[T], Tensor[T], T]()

          conv1.add(new nn.SpatialConvolution[T](192, 64, 1, 1, 1, 1, 0, 0).setInitMethod(Xavier))
          conv1.add(new nn.ReLU[T](true))

          conv3.add(new nn.SpatialConvolution[T](192, 96, 1, 1, 1, 1, 0, 0).setInitMethod(Xavier))
          conv3.add(new nn.ReLU[T](true))
          conv3.add(new nn.SpatialConvolution[T](96, 128, 3, 3, 1, 1, 1, 1).setInitMethod(Xavier))
          conv3.add(new nn.ReLU[T](true))

          conv5.add(new nn.SpatialConvolution[T](192, 16, 1, 1, 1, 1, 0, 0).setInitMethod(Xavier))
          conv5.add(new nn.ReLU[T](true))
          conv5.add(new nn.SpatialConvolution[T](16, 32, 5, 5, 1, 1, 2, 2).setInitMethod(Xavier))
          conv5.add(new nn.ReLU[T](true))

          pool.add(new nn.SpatialMaxPooling[T](3, 3, 1, 1, 1, 1).ceil())
          pool.add(new nn.SpatialConvolution[T](192, 32, 1, 1, 1, 1, 0, 0).setInitMethod(Xavier))
          pool.add(new nn.ReLU[T](true))

          concat.add(conv1)
          concat.add(conv3)
          concat.add(conv5)
          concat.add(pool)
          concat
      }
    }

    def test[T: ClassTag]()(implicit ev: TensorNumeric[T]): Unit = {
      val m1 = model[T]("mix")
      println(m1)
      val m2 = model[T]("blas")

      val m1Para = m1.parameters()
      val m2Para = m2.parameters()
      for (i <- 0 until m1Para._1.length) {
        m1Para._1(i).copy(m2Para._1(i))
      }
      val input = Tensor[T](Array(32, 192, 28, 28)).rand()
      val gradOutput = Tensor[T](Array(32, 256, 28, 28)).rand()

      val outputM1 = m1.updateOutput(input)
      val outputM2 = m2.updateOutput(input)
      outputM1 should be equals (outputM2)

      val gradInputM1 = m1.backward(input, gradOutput)
      val gradInputM2 = m2.backward(input, gradOutput)
      gradInputM1 should be equals (gradInputM2)

      Tools.averageError[T](outputM1, outputM2, "output") should be(0.0 +- 1e-5)
      Tools.averageError[T](gradInputM1, gradInputM2, "gradInput") should be(0.0 +- 1e-5)
    }

    for (i <- 0 until Tools.getRandTimes()) {
      test[Float]()
      test[Double]()
    }
  }
}
