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
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl._

import scala.util.Random

@com.intel.analytics.bigdl.tags.Serial
class SpatialCrossMapLRNSpec extends TorchSpec {
    "A SpatialCrossMapLRN Layer" should "generate correct output" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)

    val layer = new SpatialCrossMapLRN[Double](5, 1.0, 0.75, 1.0)
    val input = Tensor[Double](16, 3, 224, 224).rand()
    val output = layer.updateOutput(input)

    val code = "torch.manualSeed(" + seed + ")\n" +
      "layer = nn.SpatialCrossMapLRN(5, 1.0, 0.75, 1.0)\n" +
      "output = layer:forward(input) "

    val torchResult = TH.run(code, Map("input" -> input), Array("output"))._2
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]

    output shouldEqual luaOutput
  }

  it should "generate correct output when feature map number is large" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)

    val layer = new SpatialCrossMapLRN[Double](5, 1.0, 0.75, 1.0)
    val input = Tensor[Double](16, 32, 128, 128).rand()
    val output = layer.updateOutput(input)

    val code = "torch.manualSeed(" + seed + ")\n" +
      "layer = nn.SpatialCrossMapLRN(5, 1.0, 0.75, 1.0)\n" +
      "output = layer:forward(input) "

    val torchResult = TH.run(code, Map("input" -> input), Array("output"))._2
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]

    output shouldEqual luaOutput
  }

  it should "generate correct gradInput" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)

    val layer = new SpatialCrossMapLRN[Double](5, 1.0, 0.75, 1.0)
    val input = Tensor[Double](16, 3, 224, 224).rand()
    val gradOutput = Tensor[Double](16, 3, 224, 224).rand()
    layer.updateOutput(input)
    val output = layer.updateGradInput(input, gradOutput)

    val code = "torch.manualSeed(" + seed + ")\n" +
      "layer = nn.SpatialCrossMapLRN(5, 1.0, 0.75, 1.0)\n" +
      "layer:forward(input) " +
      "gradInput = layer:updateGradInput(input, gradOutput) "

    val torchResult = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("gradInput"))._2
    val luaOutput = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    output shouldEqual luaOutput
  }

  it should "generate correct gradInput when feature map number is large" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)

    val layer = new SpatialCrossMapLRN[Double](5, 1.0, 0.75, 1.0)
    val input = Tensor[Double](16, 32, 128, 128).rand()
    val gradOutput = Tensor[Double](16, 32, 128, 128).rand()
    layer.updateOutput(input)
    val output = layer.updateGradInput(input, gradOutput)

    val code = "torch.manualSeed(" + seed + ")\n" +
      "layer = nn.SpatialCrossMapLRN(5, 1.0, 0.75, 1.0)\n" +
      "layer:forward(input) " +
      "gradInput = layer:updateGradInput(input, gradOutput) "

    val torchResult = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("gradInput"))._2
    val luaOutput = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    output shouldEqual luaOutput
  }

  "SpatialCrossMapLRN module" should "be good in gradient check for input" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)

    val layer = new SpatialCrossMapLRN[Double](5, 1.0, 0.75, 1.0)
    val input = Tensor[Double](4, 8, 32, 32).apply1(e => Random.nextDouble())

    val checker = new GradientChecker(1e-3)
    checker.checkLayer[Double](layer, input, 1e-3) should be(true)
  }

  "SpatialCrossMapLRN module" should "be saved to or loaded from Torch model correctly" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)

    val layer = new SpatialCrossMapLRN[Double](5, 1.0, 0.75, 1.0)

    val tmpFile = java.io.File.createTempFile("module", ".t7." + suffix, TH.resultsRoot.toFile)
    val absolutePath = tmpFile.getAbsolutePath
    layer.saveTorch(absolutePath, true)

    val model = Module.loadTorch[Double](absolutePath).asInstanceOf[SpatialCrossMapLRN[Double]]
    model shouldEqual layer

    val input = Tensor[Double](16, 3, 224, 224).rand()
    val output = layer.updateOutput(input)

    val code = "torch.manualSeed(" + seed + ")\n" +
      "layer = torch.load(\'" + absolutePath + "\')\n" +
      "output = layer:forward(input) "

    val torchResult = TH.run(code, Map("input" -> input), Array("output"))._2
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]

    output shouldEqual luaOutput
  }


}
