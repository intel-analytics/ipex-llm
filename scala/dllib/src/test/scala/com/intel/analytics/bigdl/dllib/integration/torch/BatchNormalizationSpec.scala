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

package com.intel.analytics.bigdl.integration.torch

import breeze.numerics.abs
import com.intel.analytics.bigdl.nn.{BatchNormalization, GradientChecker}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl._

import scala.util.Random

@com.intel.analytics.bigdl.tags.Serial
class BatchNormalizationSpec extends TorchSpec {
    "A SpatialBatchNormalization" should "generate correct output and gradInput" in {
    torchCheck()

    val seed = 100
    RNG.setSeed(seed)

    val sbn = new BatchNormalization[Double](3, 1e-3)

    val input = Tensor[Double](16, 3)
    var i = 0
    input.apply1(e => {
      RNG.uniform(0.0, 255)
    })
    //    input.apply1(e => 0.5)
    val gradOutput = Tensor[Double](16, 3)
    i = 0
    gradOutput.apply1(e => {
      i += 1;
      0.1 * i
    })

    val gradOutput2 = Tensor[Double](16, 3)
    i = 0
    gradOutput2.apply1(e => {
      i += 1;
      0.05 * i
    })


    sbn.zeroGradParameters()
    val parameters = sbn.getParameters()._1.asInstanceOf[Tensor[Double]]
    val gradparameters = sbn.getParameters()._2.asInstanceOf[Tensor[Double]]

    val code = "torch.manualSeed(" + seed + ")\n" +
      """
        |sbn = nn.BatchNormalization(3, 1e-3)
        |sbn:zeroGradParameters()
        |local parameters, gradParameters = sbn:getParameters()
        |parameters_initial = parameters : clone()
        |gradParameters_initial = gradParameters : clone()
        |
        |output = sbn:forward(input)
        |
        |gradInput = sbn:backward(input, gradOutput)
        |
        |sbn:forward(input)
        |
        |sbn:backward(input, gradOutput2)
      """.stripMargin

    val (luaTime, torchResult) = TH.run(code,
      Map("input" -> input, "gradOutput" -> gradOutput, "gradOutput2" -> gradOutput2),
      Array("parameters_initial", "gradParameters_initial", "gradParameters",
        "output", "gradInput")
    )
    val parameterTorch = torchResult("parameters_initial").asInstanceOf[Tensor[Double]]
    val gradparameterTorch = torchResult("gradParameters_initial").asInstanceOf[Tensor[Double]]
    val gradparametersTorch = torchResult("gradParameters").asInstanceOf[Tensor[Double]]
    val outputTorch = torchResult("output").asInstanceOf[Tensor[Double]]
    val gradInputTorch = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    require(parameters == parameterTorch, "parameter compare failed")

    require(gradparameters == gradparameterTorch, "gradparameter compare failed")

    sbn.forward(input)
    sbn.backward(input, gradOutput)

    val output = sbn.forward(input)

    val gradInput = sbn.backward(input, gradOutput2)

    outputTorch.map(output, (v1, v2) => {
      assert(abs(v1 - v2) == 0)
      v1
    })

    gradInputTorch.map(gradInput, (v1, v2) => {
      assert(abs(v1 - v2) == 0)
      v1
    })

    gradparametersTorch.almostEqual(gradparameters, 1e-10)
  }

  "A SpatialBatchNormalization evaluating" should "generate correct output" in {
    torchCheck()

    val seed = 100
    RNG.setSeed(seed)

    val sbn = new BatchNormalization[Double](3, 1e-3)

    val input = Tensor[Double](16, 3)
    var i = 0
    input.apply1(e => {
      RNG.uniform(0.0, 255)
    })
    //    input.apply1(e => 0.5)
    val gradOutput = Tensor[Double](16, 3)
    i = 0
    gradOutput.apply1(e => {
      i += 1;
      0.1 * i
    })

    val gradOutput2 = Tensor[Double](16, 3)
    i = 0
    gradOutput2.apply1(e => {
      i += 1;
      0.05 * i
    })


    sbn.zeroGradParameters()
    val parameters = sbn.getParameters()._1.asInstanceOf[Tensor[Double]]
    val gradparameters = sbn.getParameters()._2.asInstanceOf[Tensor[Double]]

    val code = "torch.manualSeed(" + seed + ")\n" +
      """
        |sbn = nn.BatchNormalization(3, 1e-3)
        |sbn:zeroGradParameters()
        |local parameters, gradParameters = sbn:getParameters()
        |parameters_initial = parameters : clone()
        |gradParameters_initial = gradParameters : clone()
        |
        |output = sbn:forward(input)
        |
        |gradInput = sbn:backward(input, gradOutput)
        |
        |sbn:forward(input)
        |
        |sbn:backward(input, gradOutput2)
        |
        |sbn:evaluate()
        |
        |output = sbn:forward(input)
      """.stripMargin

    val (luaTime, torchResult) = TH.run(code,
      Map("input" -> input, "gradOutput" -> gradOutput, "gradOutput2" -> gradOutput2),
      Array("parameters_initial", "gradParameters_initial", "output"))
    val parameterTorch = torchResult("parameters_initial").asInstanceOf[Tensor[Double]]
    val gradparameterTorch = torchResult("gradParameters_initial").asInstanceOf[Tensor[Double]]
    val outputTorch = torchResult("output").asInstanceOf[Tensor[Double]]

    require(parameters == parameterTorch, "parameter compare failed")

    require(gradparameters == gradparameterTorch, "gradparameter compare failed")

    sbn.forward(input)
    sbn.backward(input, gradOutput)

    sbn.forward(input)

    sbn.backward(input, gradOutput2)

    sbn.evaluate()
    val output = sbn.forward(input)

    outputTorch.map(output, (v1, v2) => {
      assert(abs(v1 - v2) == 0)
      v1
    })

  }

  "A SpatialBatchNormalization forward backward twice" should
    "generate correct output and gradInput" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)

    val sbn = new BatchNormalization[Double](3, 1e-3)

    val input = Tensor[Double](16, 3)
    var i = 0
    input.apply1(e => {
      RNG.uniform(0.0, 255)
    })
    val gradOutput = Tensor[Double](16, 3)
    i = 0
    gradOutput.apply1(_ => Random.nextDouble())

    val gradOutput2 = Tensor[Double](16, 3)
    i = 0
    gradOutput2.apply1(_ => Random.nextDouble())


    sbn.zeroGradParameters()
    val parameters = sbn.getParameters()._1.asInstanceOf[Tensor[Double]]
    val gradparameters = sbn.getParameters()._2.asInstanceOf[Tensor[Double]]

    val code = "torch.manualSeed(" + seed + ")\n" +
      """
        |sbn = nn.BatchNormalization(3, 1e-3)
        |sbn:zeroGradParameters()
        |local parameters, gradParameters = sbn:getParameters()
        |parameters_initial = parameters : clone()
        |gradParameters_initial = gradParameters : clone()
        |
        |sbn:forward(input)
        |sbn:backward(input, gradOutput)
        |
        |output = sbn:forward(input)
        |gradInput = sbn:backward(input, gradOutput2)
      """.stripMargin

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput,
      "gradOutput2" -> gradOutput2), Array("parameters_initial", "gradParameters_initial",
      "gradParameters", "output", "gradInput"))
    val parameterTorch = torchResult("parameters_initial").asInstanceOf[Tensor[Double]]
    val gradparameterTorch = torchResult("gradParameters_initial").asInstanceOf[Tensor[Double]]
    val gradparametersTorch = torchResult("gradParameters").asInstanceOf[Tensor[Double]]
    val outputTorch = torchResult("output").asInstanceOf[Tensor[Double]]
    val gradInputTorch = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    require(parameters == parameterTorch, "parameter compare failed")

    require(gradparameters == gradparameterTorch, "gradparameter compare failed")

    sbn.forward(input)
    sbn.backward(input, gradOutput)
    val output = sbn.forward(input)
    val gradInput = sbn.backward(input, gradOutput2)

    output should be (outputTorch)
    gradInput should be (gradInputTorch)
    gradparametersTorch should be (gradparameters)

  }

  "BatchNormalization module in batch mode" should "be good in gradient check for input" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)
    val sbn = new BatchNormalization[Double](3, 1e-3)
    val input = Tensor[Double](16, 3).apply1(e => Random.nextDouble())

    val checker = new GradientChecker(1e-4)
    checker.checkLayer[Double](sbn, input, 1e-3) should be(true)
  }

  "BatchNormalization module in batch mode" should "be good in gradient check for weight" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)
    val sbn = new BatchNormalization[Double](3, 1e-3)
    val input = Tensor[Double](16, 3).apply1(e => Random.nextDouble())

    val checker = new GradientChecker(1e-4)
    checker.checkWeight[Double](sbn, input, 1e-3) should be(true)
  }

  "BatchNormalization updateGradientInput" should "generate correct output and gradInput" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)

    val sbn = new BatchNormalization[Double](3, 1e-3)

    val input = Tensor[Double](16, 3).apply1(e => {
      RNG.uniform(0.0, 255)
    })

    val gradOutput = Tensor[Double](16, 3).apply1(e => {
      RNG.uniform(0.0, 255)
    })

    sbn.zeroGradParameters()
    val parameters = sbn.getParameters()._1.asInstanceOf[Tensor[Double]]
    val gradparameters = sbn.getParameters()._2.asInstanceOf[Tensor[Double]]

    val code = "torch.manualSeed(" + seed + ")\n" +
      """
        |sbn = nn.BatchNormalization(3, 1e-3)
        |sbn:zeroGradParameters()
        |local parameters, gradParameters = sbn:getParameters()
        |parameters_initial = parameters : clone()
        |gradParameters_initial = gradParameters : clone()
        |
        |output = sbn:forward(input)
        |
        |gradInput = sbn:updateGradInput(input, gradOutput)
      """.stripMargin

    val (luaTime, torchResult) = TH.run(code,
      Map("input" -> input, "gradOutput" -> gradOutput),
      Array("parameters_initial", "gradParameters_initial", "gradParameters",
        "output", "gradInput")
    )
    val parameterTorch = torchResult("parameters_initial").asInstanceOf[Tensor[Double]]
    val gradparameterTorch = torchResult("gradParameters_initial").asInstanceOf[Tensor[Double]]
    val gradparametersTorch = torchResult("gradParameters").asInstanceOf[Tensor[Double]]
    val outputTorch = torchResult("output").asInstanceOf[Tensor[Double]]
    val gradInputTorch = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    require(parameters == parameterTorch, "parameter compare failed")

    require(gradparameters == gradparameterTorch, "gradparameter compare failed")

    val output = sbn.forward(input)

    val gradInput = sbn.updateGradInput(input, gradOutput)

    outputTorch should be (output)

    gradInputTorch should be (gradInput)

    gradparametersTorch should be (gradparameters)
  }

  "BatchNormalization updateGradientInput and acc" should "generate correct result" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)

    val sbn = new BatchNormalization[Double](3, 1e-3)

    val input = Tensor[Double](16, 3).apply1(e => {
      RNG.uniform(0.0, 255)
    })

    val gradOutput = Tensor[Double](16, 3).apply1(e => {
      RNG.uniform(0.0, 255)
    })

    sbn.zeroGradParameters()
    val parameters = sbn.getParameters()._1.asInstanceOf[Tensor[Double]]
    val gradparameters = sbn.getParameters()._2.asInstanceOf[Tensor[Double]]

    val code = "torch.manualSeed(" + seed + ")\n" +
      """
        |sbn = nn.BatchNormalization(3, 1e-3)
        |sbn:zeroGradParameters()
        |local parameters, gradParameters = sbn:getParameters()
        |parameters_initial = parameters : clone()
        |gradParameters_initial = gradParameters : clone()
        |
        |output = sbn:forward(input)
        |
        |gradInput = sbn:updateGradInput(input, gradOutput)
        |sbn:accGradParameters(input, gradOutput)
      """.stripMargin

    val (luaTime, torchResult) = TH.run(code,
      Map("input" -> input, "gradOutput" -> gradOutput),
      Array("parameters_initial", "gradParameters_initial", "gradParameters",
        "output", "gradInput")
    )
    val parameterTorch = torchResult("parameters_initial").asInstanceOf[Tensor[Double]]
    val gradparameterTorch = torchResult("gradParameters_initial").asInstanceOf[Tensor[Double]]
    val gradparametersTorch = torchResult("gradParameters").asInstanceOf[Tensor[Double]]
    val outputTorch = torchResult("output").asInstanceOf[Tensor[Double]]
    val gradInputTorch = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    require(parameters == parameterTorch, "parameter compare failed")

    require(gradparameters == gradparameterTorch, "gradparameter compare failed")

    val output = sbn.forward(input)

    val gradInput = sbn.updateGradInput(input, gradOutput)
    sbn.accGradParameters(input, gradOutput)

    outputTorch should be (output)

    gradInputTorch should be (gradInput)

    gradparametersTorch should be (gradparameters)
  }

  "BatchNormalization affine = false" should "generate correct result" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)

    val sbn = new BatchNormalization[Double](3, 1e-3, 0.1, false)

    val input = Tensor[Double](16, 3).apply1(e => {
      RNG.uniform(0.0, 255)
    })

    val gradOutput = Tensor[Double](16, 3).apply1(e => {
      RNG.uniform(0.0, 255)
    })

    val code = "torch.manualSeed(" + seed + ")\n" +
      """
        |sbn = nn.BatchNormalization(3, 1e-3, 0.1, false)
        |
        |output = sbn:forward(input)
        |
        |gradInput = sbn:updateGradInput(input, gradOutput)
        |sbn:accGradParameters(input, gradOutput)
      """.stripMargin

    val (luaTime, torchResult) = TH.run(code,
      Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput")
    )
    val outputTorch = torchResult("output").asInstanceOf[Tensor[Double]]
    val gradInputTorch = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    val output = sbn.forward(input)

    val gradInput = sbn.updateGradInput(input, gradOutput)
    sbn.accGradParameters(input, gradOutput)

    outputTorch should be (output)

    gradInputTorch should be (gradInput)
  }
}
