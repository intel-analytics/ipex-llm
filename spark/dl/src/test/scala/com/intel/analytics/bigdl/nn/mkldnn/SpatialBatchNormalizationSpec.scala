/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.nn.mkldnn

import com.intel.analytics.bigdl.mkl.{MKL, Memory}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.{Module, nn}
import org.scalatest.{FlatSpec, Matchers}

class SpatialBatchNormalizationSpec extends FlatSpec with Matchers {
  "bn updateOutput" should "work correctly" in {
    val (batchSize, channel, height, width) = (4, 64, 112, 112)
    val epsilon = 1e-5

    val initWeight = Tensor(channel).rand(-1, 1)
    val initBias = Tensor(channel).fill(0)

    val bn = SpatialBatchNormalization(channel, epsilon, initWeight = initWeight,
      initBias = initBias).setShouldConvert(true)
    val input = Tensor(batchSize, channel, height, width).rand(-1, 1)

    val output = bn.forward(input)

    val nnBn = nn.SpatialBatchNormalization(channel, epsilon,
      initWeight = initWeight, initBias = initBias)
    val nnOutput = nnBn.forward(input)

    DnnUtils.nearequals(output, nnOutput) should be (true)
  }

  "bn updateOutput multi times" should "work correctly" in {
    val (batchSize, channel, height, width) = (2, 3, 4, 4)
    val epsilon = 1e-5

    val initWeight = Tensor(channel).rand(-1, 1)
    val initBias = Tensor(channel).rand(-1, 1)

    val bn = SpatialBatchNormalization(channel, epsilon, initWeight = initWeight,
      initBias = initBias).setShouldConvert(true)
    val input = Tensor(batchSize, channel, height, width).rand(-1, 1)

    Utils.manyTimes(bn.forward(input))(10)

    val nnBn = nn.SpatialBatchNormalization(channel, epsilon,
      initWeight = initWeight, initBias = initBias)

    Utils.manyTimes(nnBn.forward(input))(10)

    DnnUtils.nearequals(bn.output.toTensor, nnBn.output.toTensor) should be (true)
  }

  "bn backward" should "work correctly" in {
    val (batchSize, channel, height, width) = (5, 64, 112, 112)
    val epsilon = 0.0f

    val initWeight = Tensor(channel).rand(-1, 1)
    val initBias = Tensor(channel).rand(-1, 1)

    val bn = SpatialBatchNormalization(channel, epsilon, initWeight = initWeight,
      initBias = initBias).setShouldConvert(true)
    val input = Tensor(batchSize, channel, height, width).rand(-1, 1)
    val gradOutput = Tensor().resizeAs(input).rand(-1, 1)

    val nnBn = nn.SpatialBatchNormalization(channel, epsilon,
      initWeight = initWeight, initBias = initBias)

    bn.forward(input)
    nnBn.forward(input)

    DnnUtils.nearequals(bn.output, nnBn.output) should be (true)
    val gradInput = bn.backward(input, gradOutput)
    val nnGradInput = nnBn.backward(input, gradOutput)

    DnnUtils.nearequals(gradInput, nnGradInput.toTensor, 1e-2) should be (true)
    DnnUtils.nearequals(bn.getParameters()._2, nnBn.getParameters()._2, 1e-2) should be (true)
  }

  "bn perf" should "work correctly" in {
    val (batchSize, channel, height, width) = (4, 64, 112, 112)
    val epsilon = 0.0f

    val initWeight = Tensor(channel).rand(-1, 1)
    val initBias = Tensor(channel).rand(-1, 1)

    val bn = SpatialBatchNormalization(channel, epsilon, initWeight = initWeight,
      initBias = initBias)
    bn.defaultFormat = Memory.Format.nChw8c
    val input = Tensor(batchSize, channel, height, width).rand(-1, 1)
    val gradOutput = Tensor().resizeAs(input).rand(-1, 1)

    val nnBn = nn.SpatialBatchNormalization(channel, epsilon,
      initWeight = initWeight, initBias = initBias)

    val times = Utils.manyTimes {
      bn.forward(input)
      bn.backward(input, gradOutput)
    } _

    val nnTimes = Utils.manyTimes {
      nnBn.forward(input)
      nnBn.backward(input, gradOutput)
    } _

    times(10)
    nnTimes(10)

    val costs = times(50)._1
    val nnCosts = nnTimes(50)._1

    println(costs)
    println(nnCosts)
  }

  "Convolution + SpatialBarchNormalization" should "work correctly" in {
    MKL.setNumThreads(1)
    val dnn = Sequential()
      .add(ConvolutionDnn(3, 64, 7, 7, 2, 2, 3, 3).setName("conv1/7x7_s2"))
      .add(SpatialBatchNormalization(64, 1e-3).setName("conv1/7x7_s2/bn"))
      .add(MemoryReOrder(inputFormat = Memory.Format.any, outputFormat = Memory.Format.nchw))

    val blas = Sequential()
      .add(nn.SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3).setName("conv1/7x7_s2"))
      .add(nn.SpatialBatchNormalization(64, 1e-3).setName("conv1/7x7_s2/bn"))
      .add(nn.Identity())

//    for (i <- dnn.parameters()._1.indices) {
//      blas.parameters()._1(i).rand(-1, 1)
//      dnn.parameters()._1(i).copy(blas.parameters()._1(i))
//    }

    val input = Tensor(4, 3, 224, 224).rand()

    blas.forward(input)
    dnn.forward(input)

//    DnnUtils.nearequals(dnn.output.toTensor, blas.output.toTensor)

    val gradOutput = Tensor().resizeAs(blas.output.toTensor).rand()

    blas.backward(input, gradOutput)
    dnn.backward(input, gradOutput)

//    DnnUtils.nearequals(dnn.gradInput.toTensor, blas.gradInput.toTensor)

    blas.resetTimes()
    dnn.resetTimes()

    val blasCost = Utils.manyTimes {
      blas.forward(input)
      blas.backward(input, gradOutput)
    }(10)._1

    val dnnCost = Utils.manyTimes {
      dnn.forward(input)
      dnn.backward(input, gradOutput)
    }(10)._1

    println(blasCost)
    println(dnnCost)

    def format(v: Double): Double = {
      (v / 1e6 / 10).formatted("%2.4f").toDouble
    }
    val names = blas.getTimes().map(_._1.getName())
    val blasForwardTime = blas.getTimes().map(x => format(x._2))
    val blasBackwardTime = blas.getTimes().map(x => format(x._3))

    val dnnForwardTime = dnn.getTimes().map(x => format(x._2))
    val dnnBackwardTime = dnn.getTimes().map(x => format(x._3))

    val forwardUpgrade = blasForwardTime.zip(dnnForwardTime).map { t =>
      ((t._1 - t._2) / t._2.toDouble).formatted("%2.2f")
    }
    val backwardUpgrade = blasBackwardTime.zip(dnnBackwardTime).map { t =>
      ((t._1 - t._2) / t._2.toDouble).formatted("%2.2f")
    }

    val header = List("MODULE NAME", "MKL-BLAS", "MKL-DNN", "UPGRADE")

    def rows4(input: List[Array[_]]): List[List[_]] = {
      input(0).toList zip input(1).toList zip input(2) zip input(3) map {
        case (((a, b), c), d) => List(a, b, c, d)
      }
    }

    val forwardTime = rows4(List(names, blasForwardTime, dnnForwardTime, forwardUpgrade))

    val backwardTime = rows4(List(names, blasBackwardTime, dnnBackwardTime, backwardUpgrade))

    println(Tabulator.format(header:: forwardTime))
    println("=" * 80)
    println(Tabulator.format(header:: backwardTime))

    forwardUpgrade.foreach {x =>
      if (x.contains("conv1")) {
        x.toDouble should be > 0.0
      }
    }

    backwardUpgrade.foreach {x =>
      if (x.contains("conv1")) {
        x.toDouble should be > 0.0
      }
    }
  }

  "bn + linear" should "work correctly" in {
    val batch = 4
    val channel = 64
    val height = 16
    val width = 16
    val channel2 = channel * height * width
    val channel3 = 1000
    val input = Tensor[Float](Array(batch, channel, height, width)).rand()
    val initWeight1 = Tensor[Float](Array(channel)).rand()
    val initBias1 = Tensor[Float](Array(channel)).rand()
    val initWeight2 = Tensor[Float](Array(channel3, channel2)).rand()
    val initBias2 = Tensor[Float](Array(channel3)).rand()

    val bn1 = SpatialBatchNormalization(64, initWeight = initWeight1, initBias = initBias1)
        .setShouldConvert(false)
    bn1.defaultFormat = Memory.Format.nChw8c
    val seq = Sequential()
      .add(bn1)
      .add(SpatialBatchNormalization(64, initWeight = initWeight1, initBias = initBias1)
        .setShouldConvert(false))
      .add(Linear(channel2, channel3, initWeight = initWeight2, initBias = initBias2)
        .setShouldConvert(false))
      .add(ReLUDnn())

    seq.forward(input)
    println("=" * 80)

    val gradOutput = Tensor[Float]().resizeAs(seq.output.toTensor).rand()
    seq.backward(input, gradOutput)

    val seq2 = Sequential()
      .add(nn.SpatialBatchNormalization(64, initWeight = initWeight1, initBias = initBias1))
      .add(nn.SpatialBatchNormalization(64, initWeight = initWeight1, initBias = initBias1))
      .add(nn.View(Array(batch, channel2)))
      .add(nn.Linear(channel2, channel3, initWeight = initWeight2, initBias = initBias2))
      .add(nn.ReLU())

    seq2.forward(input)

    val t1 = Utils.manyTimes {
      seq.forward(input)
      seq.backward(input, gradOutput)
    } _

    val t2 = Utils.manyTimes {
      seq2.forward(input)
      seq2.backward(input, gradOutput)
    } _

    t1(10)
    t2(10)

    seq.resetTimes()
    seq2.resetTimes()

    val dnnTime = t1(10)._1
    val nnTime = t2(10)._1
    println(dnnTime)
    println(nnTime)
    println(seq.getTimes().mkString("\n"))
    println(seq2.getTimes().mkString("\n"))

    dnnTime should be < nnTime
  }

  "linear + linear, the first linear with a 4-D input" should "work correctly" in {
    val inputSize = 16 * 16 * 16
    val outputSize = 16 * 16 * 16
    val initWeight = Tensor[Float](outputSize, inputSize).rand()
    val initBias = Tensor[Float](outputSize)

    val seq = Sequential()
      .add(Linear(outputSize, inputSize, initWeight = initWeight, initBias = initBias)
        .setShouldConvert(false))
      .add(Linear(outputSize, inputSize, initWeight = initWeight, initBias = initBias))

    val input = Tensor[Float](16, inputSize).rand()

    seq.forward(input)
    seq.backward(input, input)

    val input2 = Tensor[Float](16, 16, 16, 16).rand()
    seq.forward(input2)
    seq.backward(input2, input)
  }

  "bn clone" should "work correctly" in {
    val (batchSize, channel, height, width) = (2, 3, 4, 4)
    val epsilon = 1e-5

    val initWeight = Tensor(channel).rand()
    val initBias = Tensor(channel).rand()

    val bn = SpatialBatchNormalization(channel, epsilon, initWeight = initWeight,
      initBias = initBias)
    val input = Tensor(batchSize, channel, height, width).rand()

    bn.cloneModule().forward(input).toTensor should be (bn.forward(input).toTensor)
  }

  "bn + conv + relu" should "work correctly" in {
    val inputChannel = 64
    val outputChannel = 64
    val stride = 3
    val input = Tensor(4, 64, 57, 57).rand(-1, 1)

    val dnn = Sequential()
      .add(ReLUDnn())
      .add(ConvolutionDnn(inputChannel, outputChannel, 1, 1, 1, 1, 0, 0))
      .add(SpatialBatchNormalization(outputChannel))
      .add(ReLUDnn(true))
      .add(ConvolutionDnn(outputChannel, outputChannel, 3, 3, stride, stride, 1, 1))
      .add(SpatialBatchNormalization(outputChannel))
      .add(ReLUDnn(true))
      .add(ConvolutionDnn(outputChannel, outputChannel*4, 1, 1, 1, 1, 0, 0))
      .add(SpatialBatchNormalization(outputChannel * 4))

    val blas = Sequential()
      .add(nn.ReLU())
      .add(nn.SpatialConvolution(inputChannel, outputChannel, 1, 1, 1, 1, 0, 0))
      .add(nn.SpatialBatchNormalization(outputChannel))
      .add(nn.ReLU(true))
      .add(nn.SpatialConvolution(outputChannel, outputChannel, 3, 3, stride, stride, 1, 1))
      .add(nn.SpatialBatchNormalization(outputChannel))
      .add(nn.ReLU(true))
      .add(nn.SpatialConvolution(outputChannel, outputChannel*4, 1, 1, 1, 1, 0, 0))
      .add(nn.SpatialBatchNormalization(outputChannel * 4))

    blas.forward(input)
    dnn.forward(input)
    val gradOutput = Tensor().resizeAs(blas.output.toTensor).rand(-1, 1)
    blas.backward(input, gradOutput)
    dnn.backward(input, gradOutput)

    val dnnTime = Utils.manyTimes {
      dnn.forward(input)
      dnn.backward(input, gradOutput)
    } _

    val blasTime = Utils.manyTimes {
      blas.forward(input)
      blas.backward(input, gradOutput)
    } _

    val warm = 10
    val iter = 20

    dnnTime(warm)
    blasTime(iter)

    dnn.resetTimes()
    blas.resetTimes()

    dnnTime(iter)
    blasTime(iter)

    compare(dnn, blas)
  }

  def compare(dnn: Module[Float], blas: Module[Float]): Unit = {
    def format(v: Double): Double = {
      (v / 1e6 / 10).formatted("%2.4f").toDouble
    }
    val names = blas.getTimes().map(_._1.getName())
    val blasForwardTime = blas.getTimes().map(x => format(x._2))
    val blasBackwardTime = blas.getTimes().map(x => format(x._3))

    val dnnForwardTime = dnn.getTimes().map(x => format(x._2))
    val dnnBackwardTime = dnn.getTimes().map(x => format(x._3))

    val forwardUpgrade = blasForwardTime.zip(dnnForwardTime).map { t =>
      ((t._1 - t._2) / t._1.toDouble).formatted("%2.2f")
    }
    val backwardUpgrade = blasBackwardTime.zip(dnnBackwardTime).map { t =>
      ((t._1 - t._2) / t._1.toDouble).formatted("%2.2f")
    }

    val header = List("MODULE NAME", "MKL-BLAS", "MKL-DNN", "UPGRADE")

    def rows4(input: List[Array[_]]): List[List[_]] = {
      input(0).toList zip input(1).toList zip input(2) zip input(3) map {
        case (((a, b), c), d) => List(a, b, c, d)
      }
    }

    val forwardTime = rows4(List(names, blasForwardTime, dnnForwardTime, forwardUpgrade))

    val backwardTime = rows4(List(names, blasBackwardTime, dnnBackwardTime, backwardUpgrade))

    println(Tabulator.format(header:: forwardTime))
    println("=" * 80)
    println(Tabulator.format(header:: backwardTime))
  }

  "bn with dynamic input size" should "work correctly" in {
    val (channel, height, width) = (3, 4, 4)
    val epsilon = 1e-5

    val initWeight = Tensor(channel).rand(-1, 1)
    val initBias = Tensor(channel).fill(0)

    val bn = SpatialBatchNormalization(channel, epsilon, initWeight = initWeight,
      initBias = initBias)
    val nnBn = nn.SpatialBatchNormalization(channel, epsilon,
      initWeight = initWeight, initBias = initBias)

    for (batchSize <- Array(2, 3, 4, 2)) {
      val input = Tensor(batchSize, channel, height, width).rand(-1, 1)
      val (weight1, gradweight1) = bn.getParameters()
      val (weight2, gradweight2) = nnBn.getParameters()

      bn.zeroGradParameters()
      nnBn.zeroGradParameters()

      bn.forward(input)
      nnBn.forward(input)
      DnnUtils.nearequals(bn.output, nnBn.output) should be(true)

      val gradOutput = Tensor().resizeAs(input).rand()

      bn.backward(input, gradOutput)
      nnBn.backward(input, gradOutput)

      DnnUtils.nearequals(weight1, weight2) should be(true)
      DnnUtils.nearequals(gradweight1, gradweight2) should be(true)

      DnnUtils.nearequals(bn.gradInput, nnBn.gradInput) should be(true)

      println("=" * 120)
    }
  }

  "bn with dynamic input size 1111" should "work correctly" in {
    val (channel, height, width) = (64, 112, 112)
    val epsilon = 1e-3

    val initWeight = Tensor(channel).rand(-1, 1)
    val initBias = Tensor(channel).rand(-1, 1)

    val bn = SpatialBatchNormalization(channel, epsilon, initWeight = initWeight,
      initBias = initBias)
    val nnBn = nn.SpatialBatchNormalization(channel, epsilon,
      initWeight = initWeight, initBias = initBias)

    for (batchSize <- Array(2, 3, 4, 2)) {
      bn.zeroGradParameters()
      nnBn.zeroGradParameters()

      val (weight, gradWeight) = bn.getParameters()
      val (nnWeight, nnGradWeight) = nnBn.getParameters()

      val input = Tensor(batchSize, channel, height, width).rand(-1, 1)
      val gradOutput = Tensor().resizeAs(input).rand(-1, 1)

      bn.forward(input)
      nnBn.forward(input)

      DnnUtils.nearequals(bn.output, nnBn.output) should be (true)

      val gradInput = bn.backward(input, gradOutput)
      val nnGradInput = nnBn.backward(input, gradOutput)

      DnnUtils.nearequals(gradInput, nnGradInput.toTensor) should be (true)
      DnnUtils.nearequals(bn.getParameters()._2, nnBn.getParameters()._2, 1e-3) should be (true)
    }
  }

  "bn with conv input" should "work correctly" in {
    val (channel, height, width) = (64, 112, 112)
    val epsilon = 1e-3
    val batchSize = 2

    RNG.setSeed(100)
    val input = Tensor[Float](Array(batchSize, 64, 112, 112)).rand(-1, 1)
    RNG.setSeed(100)
    val initWeight = Tensor(channel).rand(-1, 1)
    val initBias = Tensor(channel).fill(0f)
    val bn = SpatialBatchNormalization(channel, epsilon, initWeight = initWeight,
      initBias = initBias)
    RNG.setSeed(100)
    val nnBn = nn.SpatialBatchNormalization(channel, epsilon, initWeight = initWeight,
      initBias = initBias)

    bn.zeroGradParameters()
    nnBn.zeroGradParameters()

    val (weight, gradWeight) = bn.getParameters()
    val (nnWeight, nnGradWeight) = nnBn.getParameters()
    DnnUtils.nearequals(weight, nnWeight) should be(true)
    DnnUtils.nearequals(gradWeight, nnGradWeight, 1e-3) should be(true)

    // val input = Tensor(batchSize, channel, height, width).rand(-1, 1)
    val gradOutput = Tensor().resizeAs(input).copy(input)

    val out1 = bn.forward(input)
    val out2 = nnBn.forward(input)

    out1.storage()

    bn.output.almostEqual(nnBn.output, 1e-5)
//    DnnUtils.nearequals(bn.output, nnBn.output) should be (true)

    val gradInput = bn.backward(input, gradOutput)
    val nnGradInput = nnBn.backward(input, gradOutput)

    bn.getParameters()._2.almostEqual(nnBn.getParameters()._2, 1e-5)
    DnnUtils.getunequals(gradInput, nnGradInput.toTensor) should be (true)
    DnnUtils.nearequals(bn.getParameters()._2, nnBn.getParameters()._2, 1e-4) should be (true)

    DnnUtils.nearequals(weight, nnWeight) should be(true)
    DnnUtils.nearequals(gradWeight, nnGradWeight, 1e-4) should be(true)
  }

  "A nChw8c input" should "work correctly" in {
    val (batchSize, channel, height, width) = (2, 256, 56, 56)
    val input = Tensor[Float](batchSize, channel, height, width).rand(-1, 1)
    val gradOutput = Tensor[Float](batchSize, channel, height, width).rand(-1, 1)
    val dnn = Sequential()
      .add(MemoryReOrder(5, 8))
      .add(SpatialBatchNormalization(channel, 1e-3))
      .add(MemoryReOrder(8, 5))
    val blas = Sequential().add(nn.SpatialBatchNormalization(channel, 1e-3))

    dnn.getParameters()._1.copy(blas.getParameters()._1)

    dnn.forward(input)
    blas.forward(input)

    dnn.backward(input, gradOutput)
    blas.backward(input, gradOutput)

    DnnUtils.nearequals(dnn.output.toTensor, blas.output.toTensor, 1e-4) should be (true)
    DnnUtils.nearequals(dnn.gradInput.toTensor, blas.gradInput.toTensor, 1e-4) should be (true)
    DnnUtils.nearequals(dnn.getParameters()._2, blas.getParameters()._2, 1e-3) should be (true)
  }

  "A nChw16c input" should "work correctly" in {
    val (batchSize, channel, height, width) = (2, 256, 56, 56)
    val input = Tensor[Float](batchSize, channel, height, width).rand(-1, 1)
    val gradOutput = Tensor[Float](batchSize, channel, height, width).rand(-1, 1)
    val dnn = Sequential()
      .add(MemoryReOrder(5, 9))
      .add(SpatialBatchNormalization(channel, 1e-3))
      .add(MemoryReOrder(9, 5))
    val blas = Sequential().add(nn.SpatialBatchNormalization(channel, 1e-3))

    dnn.getParameters()._1.copy(blas.getParameters()._1)

    dnn.forward(input)
    blas.forward(input)

    dnn.backward(input, gradOutput)
    blas.backward(input, gradOutput)

    DnnUtils.nearequals(dnn.output.toTensor, blas.output.toTensor) should be (true)
    DnnUtils.nearequals(dnn.gradInput.toTensor, blas.gradInput.toTensor) should be (true)
    DnnUtils.nearequals(dnn.getParameters()._2, blas.getParameters()._2, 1e-3) should be (true)
  }

  "A simple input with java serialization" should "work correctly" in {
    val (batchSize, channel, height, width) = (4, 64, 112, 112)
    val epsilon = 1e-5

    val initWeight = Tensor(channel).rand(-1, 1)
    val initBias = Tensor(channel).fill(0)
    val input = Tensor(batchSize, channel, height, width).rand(-1, 1)

    val bn = SpatialBatchNormalization(channel, epsilon, initWeight = initWeight,
      initBias = initBias).setShouldConvert(true)
    bn.forward(input)

    val bnClone = bn.cloneModule().asInstanceOf[SpatialBatchNormalization[Float]]

    bn.runningMean should be (bnClone.runningMean)
    bn.runningVar should be (bnClone.runningVar)
  }

//  "A nchw input with infer" should "work correctly" in {
//    // BUG: This test can pass if and only if the unbiased variance = std / frameSize.
//    //      The BigDL/Torch7 version is unbiased variance = std / (frameSize - 1).
//    //      But if we extend the input from 1x1x2x2 to 4x256x56x56, the result can be tolerated.
//    RNG.setSeed(1)
//    val (batchSize, channel, height, width) = (4, 256, 56, 56)
//    val input = Tensor[Float](batchSize, channel, height, width).rand(-1, 1)
//    val gradOutput = Tensor[Float](batchSize, channel, height, width).rand(-1, 1)
//    val initWeight = Tensor[Float](channel).rand(-1, 1)
//    val initBias = Tensor[Float](channel).rand(-1, 1)
//    val dnn = Sequential()
//      .add(MemoryReOrder(5, 5))
//      .add(SpatialBatchNormalization(channel, 1e-3, momentum = 0.1, initWeight = initWeight,
//        initBias = initBias))
//      .add(MemoryReOrder(5, 5).setShouldConvert(true))
//    val blas = Sequential().add(nn.SpatialBatchNormalization(channel, 1e-3, momentum = 0.1,
//      initWeight = initWeight, initBias = initBias))
//
//    dnn.getParameters()._1.copy(blas.getParameters()._1)
//
//    dnn.training()
//    blas.training()
//    for (i <- 0 until 4) {
//      dnn.zeroGradParameters()
//      blas.zeroGradParameters()
//      input.rand(-1, 1)
//      dnn.forward(input)
//      blas.forward(input)
//      dnn.backward(input, gradOutput)
//      blas.backward(input, gradOutput)
//
//      DnnUtils.nearequals(dnn.output.toTensor, blas.output.toTensor, 1e-4) should be(true)
//      DnnUtils.nearequals(dnn.gradInput.toTensor, blas.gradInput.toTensor, 1e-4) should be(true)
//      DnnUtils.nearequals(dnn.getParameters()._2, blas.getParameters()._2, 1e-3) should be(true)
//    }
//
//    input.rand(-1, 1)
//    blas.evaluate()
//    blas.forward(input)
//
//    dnn.evaluate()
//    dnn.forward(input)
//
//    DnnUtils.nearequals(dnn.output.toTensor, blas.output.toTensor, 1e-4) should be (true)
//  }

  "Sbn with relu fusion" should "work correctly" in {
    val (batchSize, channel, height, width) = (4, 64, 112, 112)
    val epsilon = 1e-5

    val initWeight = Tensor(channel).rand(-1, 1)
    val initBias = Tensor(channel).fill(0)

    val bn1 = SpatialBatchNormalization(channel, epsilon, initWeight = initWeight,
      initBias = initBias).setShouldConvert(true)
    val bn2 = SpatialBatchNormalization(channel, epsilon, initWeight = initWeight,
      initBias = initBias).setShouldConvert(true)
    val input = Tensor(batchSize, channel, height, width).rand(-1, 1)

    val model = Sequential().add(bn1).add(ReLUDnn())

    model.evaluate()
    model.forward(input)

    bn2.relu = true
    bn2.evaluate()
    bn2.forward(input)

    model.output.toTensor.storage()
    model.output should be (bn2.output)
  }
}
