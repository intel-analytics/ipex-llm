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
package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.apache.spark.SparkContext
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.util.Random

class SpatialBatchNormalizationSpec extends FlatSpec with Matchers with BeforeAndAfter {
  before {
    System.setProperty("bigdl.localMode", "true")
    System.setProperty("spark.master", "local[2]")
    Engine.init
  }

  after {
    System.clearProperty("bigdl.localMode")
    System.clearProperty("spark.master")
  }

  "SpatialBacthNormalization parameter sync" should "work properly" in {
    Engine.init

    val bn = SpatialBatchNormalization[Float](2)
    bn.setParallism(1)
    bn.weight.fill(1.0f)
    bn.bias.fill(1.0f)

    val input = Tensor[Float](2, 2, 1, 1)
    input.select(1, 1).fill(1.0f)
    input.select(1, 2).fill(2.0f)

    val gradOutput = Tensor[Float](2, 2, 1, 1)

    gradOutput.select(1, 1).fill(2.0f)
    gradOutput.select(1, 2).fill(1.0f)

    val output = bn.forward(input)

    val gradInput = bn.backward(input, gradOutput)

    val saveMean = bn.saveMean
    val saveStd = bn.saveStd
    val runningMean = bn.runningMean
    val runningVar = bn.runningVar

    val bn1 = SpatialBatchNormalization[Float](2)
    bn1.setParallism(2)

    bn1.weight.fill(1.0f)
    bn1.bias.fill(1.0f)

    val bn2 = bn1.cloneModule().asInstanceOf[BatchNormalization[Float]]

    val modules = Array(bn1, bn2)

    val input1 = Tensor[Float](1, 2, 1, 1).fill(1.0f)

    val input2 = Tensor[Float](1, 2, 1, 1).fill(2.0f)

    val inputs = Array(input1, input2)

    val gradOutput1 = Tensor[Float](1, 2, 1, 1).fill(2.0f)
    val gradOutput2 = Tensor[Float](1, 2, 1, 1).fill(1.0f)

    val gradOutputs = Array(gradOutput1, gradOutput2)

    Engine.default.invokeAndWait2((0 until modules.size).map(i =>
      () => {
        val trainStart = System.nanoTime()
        val sub = modules(i)
        val subInput = inputs(i)
        val subGradOutput = gradOutputs(i)
        sub.forward(subInput)
        sub.backward(subInput, subGradOutput)
      }
    ))

    val saveMean1 = bn1.saveMean
    val saveStd1 = bn1.saveStd
    val runningMean1 = bn1.runningMean
    val runningVar1 = bn1.runningVar
    val gradInput1 = bn1.gradInput
    val out1 = bn1.output.squeeze

    val saveMean2 = bn2.saveMean
    val saveStd2 = bn2.saveStd
    val runningMean2 = bn2.runningMean
    val runningVar2 = bn2.runningVar
    val gradInput2 = bn2.gradInput
    val out2 = bn2.output.squeeze()

    saveMean should be (saveMean1)
    saveMean should be (saveMean2)
    saveStd should be (saveStd1)
    saveStd should be (saveStd2)
    runningMean should be (runningMean1)
    runningMean should be (runningMean2)
    runningVar should be (runningVar1)
    runningVar should be (runningVar2)

    val sout1 = output.select(1, 1).squeeze()
    sout1  should be (out1)

    val sout2 = output.select(1, 2).squeeze()
    sout2 should be (bn2.output)

    val gin1 = gradInput.select(1, 1)

    val gin2 = gradInput.select(1, 2)

    gin1.squeeze should be (gradInput1.squeeze)
    gin2.squeeze should be (gradInput2.squeeze)
  }

  "SpatialBatchNormalization module in batch mode" should "be good in gradient check " +
    "for input" in {
    val seed = 100
    RNG.setSeed(seed)
    val sbn = new SpatialBatchNormalization[Double](3, 1e-3)
    val input = Tensor[Double](16, 3, 4, 4).apply1(e => Random.nextDouble())

    val checker = new GradientChecker(1e-4)
    checker.checkLayer[Double](sbn, input, 1e-3) should be(true)
  }

  "SpatialBatchNormalization backward" should "be good when affine is false" in {
    val layer = SpatialBatchNormalization[Float](3, affine = false)
    val input = Tensor[Float](4, 3, 24, 24).fill(1)
    val gradOutput = Tensor[Float](4, 3, 24, 24).fill(1)
    val output = layer.forward(input)
    output should be(Tensor[Float](4, 3, 24, 24).fill(0))
    val gradInput = layer.backward(input, gradOutput)
    gradInput should be(Tensor[Float](4, 3, 24, 24).fill(0))
  }

  "SpatialBatchNormalization module in batch mode" should "be good in gradient check " +
    "for weight" in {
    val seed = 100
    RNG.setSeed(seed)
    val sbn = new SpatialBatchNormalization[Double](3, 1e-3)
    val input = Tensor[Double](16, 3, 4, 4).apply1(e => Random.nextDouble())

    val checker = new GradientChecker(1e-4)
    checker.checkWeight[Double](sbn, input, 1e-3) should be(true)
  }

  "A SpatialBatchNormalization" should "generate same output for NHWC and NCHW" in {
    val inputNCHW = Tensor[Float](4, 256, 8, 8).rand()
    val inputNHWC = inputNCHW.transpose(2, 4).transpose(2, 3).contiguous()
    val weight = Tensor[Float](256).rand()
    val bias = Tensor[Float](256).rand()
    val bnNCHW = SpatialBatchNormalization[Float](nOutput = 256, initWeight = weight,
      initBias = bias)
    val bnNHWC = SpatialBatchNormalization[Float](nOutput = 256, dataFormat = DataFormat.NHWC,
      initWeight = weight, initBias = bias)
    val outputNCHW = bnNCHW.forward(inputNCHW)
    val outputNHWC = bnNHWC.forward(inputNHWC)
    outputNCHW.almostEqual(outputNHWC.transpose(2, 4).transpose(3, 4), 1e-5) should be(true)
  }

  "A SpatialBatchNormalization update gradinput" should
    "generate same output for NHWC and NCHW" in {
    val inputNCHW = Tensor[Float](4, 256, 8, 8).rand()
    val inputNHWC = inputNCHW.transpose(2, 4).transpose(2, 3).contiguous()
    val gradientNCHW = Tensor[Float](4, 256, 8, 8).rand()
    val gradientNHWC = gradientNCHW.transpose(2, 4).transpose(2, 3).contiguous()
    val weight = Tensor[Float](256).rand()
    val bias = Tensor[Float](256).rand()
    val bnNCHW = SpatialBatchNormalization[Float](nOutput = 256, initWeight = weight,
      initBias = bias)
    val bnNHWC = SpatialBatchNormalization[Float](nOutput = 256, dataFormat = DataFormat.NHWC,
      initWeight = weight, initBias = bias)
    val outputNCHW = bnNCHW.forward(inputNCHW)
    val outputNHWC = bnNHWC.forward(inputNHWC)

    val backpropNCHW = bnNCHW.updateGradInput(inputNCHW, gradientNCHW)
    val backpropNHWC = bnNHWC.updateGradInput(inputNHWC, gradientNHWC)

    backpropNCHW.almostEqual(backpropNHWC.transpose(2, 4).transpose(3, 4), 1e-5) should be(true)
  }

  "A SpatialBatchNormalization acc gradient" should "generate same output for NHWC and NCHW" in {
    val inputNCHW = Tensor[Float](4, 256, 8, 8).rand()
    val inputNHWC = inputNCHW.transpose(2, 4).transpose(2, 3).contiguous()
    val gradientNCHW = Tensor[Float](4, 256, 8, 8).rand()
    val gradientNHWC = gradientNCHW.transpose(2, 4).transpose(2, 3).contiguous()
    val weight = Tensor[Float](256).rand()
    val bias = Tensor[Float](256).rand()
    val bnNCHW = SpatialBatchNormalization[Float](nOutput = 256, initWeight = weight,
      initBias = bias)
    val bnNHWC = SpatialBatchNormalization[Float](nOutput = 256, dataFormat = DataFormat.NHWC,
      initWeight = weight, initBias = bias)
    val outputNCHW = bnNCHW.forward(inputNCHW)
    val outputNHWC = bnNHWC.forward(inputNHWC)

    bnNCHW.backward(inputNCHW, gradientNCHW)
    bnNHWC.backward(inputNHWC, gradientNHWC)

    bnNCHW.gradWeight.almostEqual(bnNHWC.gradWeight, 1e-5)
    bnNCHW.gradBias.almostEqual(bnNHWC.gradBias, 1e-5)
  }

  "bn with NHWC" should "return correct extra parameters" in {
    val bn1 = SpatialBatchNormalization[Float](16, dataFormat = DataFormat.NCHW)
    val bn2 = SpatialBatchNormalization[Float](16, dataFormat = DataFormat.NHWC)

    bn2.parameters()._1.zip(bn1.parameters()._1).foreach {
      case (bn2Para, bn1Para) => bn2Para.copy(bn1Para)
    }

    val input1 = Tensor[Float](4, 16, 3, 3).rand(-1, 1)
    bn1.forward(input1)

    val input2 = input1.transpose(2, 4).contiguous()
    bn2.forward(input2)

    bn1.getExtraParameter().zip(bn2.getExtraParameter()).foreach {
      case (p1, p2) =>
        p1.size() should be (p2.size())
        p1 should be (p2)
    }
  }
}

class SpatialBatchNormalizationSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val spatialBatchNorm = SpatialBatchNormalization[Float](5).
      setName("spatialBatchNorm")
    val input = Tensor[Float](2, 5, 4, 5).apply1(_ => Random.nextFloat())
    runSerializationTest(spatialBatchNorm, input)
  }
}
