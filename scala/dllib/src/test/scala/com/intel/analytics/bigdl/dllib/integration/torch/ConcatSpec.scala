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

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._

import scala.math._

@com.intel.analytics.bigdl.tags.Serial
class ConcatSpec extends TorchSpec {
    "A Concat Container with Linear" should "generate correct output and grad " in {
    torchCheck()
    val seed = 2
    RNG.setSeed(seed)
    val module = new Concat[Double](2)
    val layer1 = new Sequential[Double]()
    val layer2 = new Sequential[Double]()
    layer1.add(new SpatialBatchNormalization[Double](3, 1e-3))
    layer2.add(new SpatialBatchNormalization[Double](3, 1e-3))
    module.add(layer1).add(layer2)

    val input = Tensor[Double](4, 3, 4, 4).randn()
    val gradOutput = Tensor[Double](4, 6, 4, 4).randn()

    val code = "torch.manualSeed(" + seed + ")\n" +
      """
    module = nn.Concat(2)
    layer1 = nn.Sequential()
    layer2 = nn.Sequential()
    layer1:add(nn.SpatialBatchNormalization(3, 1e-3))
    layer2:add(nn.SpatialBatchNormalization(3, 1e-3))
    module:add(layer1):add(layer2)
    local parameters, gradParameters = module:getParameters()
    module:zeroGradParameters()
    parameters_initial = parameters : clone()
    gradParameters_initial = gradParameters : clone()

    output = module:forward(input)
    gradInput = module:backward(input,gradOutput)
      """

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput", "module", "parameters_initial", "gradParameters_initial"))
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val gradParametersInitial = torchResult("gradParameters_initial").asInstanceOf[Tensor[Double]]
    val parametersInitial = torchResult("parameters_initial").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]
    val luaModule = torchResult("module")
      .asInstanceOf[Module[Double]]

    val (parameters, gradParameters) = module.getParameters()
    require(gradParametersInitial == gradParameters)
    require(parametersInitial == parameters)

    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    luaOutput.almostEqual(output, 1e-15)
    luaGradInput.almostEqual(gradInput, 1e-15)

    println("Test case : Concat, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "A Concat Container updateGradInput and acc with Linear" should
    "generate correct output and grad " in {
    torchCheck()
    val seed = 2
    RNG.setSeed(seed)
    val module = new Concat[Double](2)
    val layer1 = new Sequential[Double]()
    val layer2 = new Sequential[Double]()
    layer1.add(new SpatialBatchNormalization[Double](3, 1e-3))
    layer2.add(new SpatialBatchNormalization[Double](3, 1e-3))
    module.add(layer1).add(layer2)

    val input = Tensor[Double](4, 3, 4, 4).randn()
    val gradOutput = Tensor[Double](4, 6, 4, 4).randn()

    val code = "torch.manualSeed(" + seed + ")\n" +
      """
    module = nn.Concat(2)
    layer1 = nn.Sequential()
    layer2 = nn.Sequential()
    layer1:add(nn.SpatialBatchNormalization(3, 1e-3))
    layer2:add(nn.SpatialBatchNormalization(3, 1e-3))
    module:add(layer1):add(layer2)
    local parameters, gradParameters = module:getParameters()
    module:zeroGradParameters()
    parameters_initial = parameters : clone()
    gradParameters_initial = gradParameters : clone()

    output = module:forward(input)
    gradInput = module:backward(input,gradOutput)
      """

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput", "module", "parameters_initial", "gradParameters_initial",
      "parameters", "gradParameters"))
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val gradParametersInitial = torchResult("gradParameters_initial").asInstanceOf[Tensor[Double]]
    val parametersInitial = torchResult("parameters_initial").asInstanceOf[Tensor[Double]]
    val gradParametersLua = torchResult("gradParameters").asInstanceOf[Tensor[Double]]
    val parametersLua = torchResult("parameters").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]
    val luaModule = torchResult("module")
      .asInstanceOf[Module[Double]]

    val (parameters, gradParameters) = module.getParameters()
    require(gradParametersInitial == gradParameters)
    require(parametersInitial == parameters)

    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.updateGradInput(input, gradOutput)
    module.accGradParameters(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    luaOutput.almostEqual(output, 1e-15)
    luaGradInput.almostEqual(gradInput, 1e-15)
    gradParametersLua.almostEqual(gradParameters, 1e-11)
    parametersLua.almostEqual(parameters, 1e-11)
    println("Test case : Concat, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "A Concat Container" should "generate correct output and grad" in {
    torchCheck()
    val module = new Concat[Double](2)
    val layer1 = new Sequential[Double]()
    val layer2 = new Sequential[Double]()
    layer1.add(new LogSoftMax())
    layer2.add(new LogSoftMax())
    module.add(layer1).add(layer2)

    val input = Tensor[Double](4, 1000).randn()
    val gradOutput = Tensor[Double](4, 2000).randn()

    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code =
      """
    module = nn.Concat(2)
    layer1 = nn.Sequential()
    layer2 = nn.Sequential()
    layer1:add(nn.LogSoftMax())
    layer2:add(nn.LogSoftMax())
    module:add(layer1):add(layer2)
    module:zeroGradParameters()

    output = module:forward(input)
    gradInput = module:backward(input,gradOutput)
      """

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput", "module"))
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]
    val luaModule = torchResult("module")
      .asInstanceOf[Module[Double]]

    luaOutput should be(output)
    luaGradInput should be(gradInput)

    println("Test case : Concat, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }
}
