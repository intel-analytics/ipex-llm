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

import com.intel.analytics.bigdl.nn.{Sequential, SpatialDivisiveNormalization}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._

import scala.util.Random

@com.intel.analytics.bigdl.tags.Serial
class SpatialDivisiveNormalizationSpec extends TorchSpec {
    "A SpatialDivisiveNormalization 3D input" should "generate correct output and grad" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)

    val layer = new SpatialDivisiveNormalization[Double]()
    val model = new Sequential[Double]()
    model.add(layer)

    Random.setSeed(3)
    val input = Tensor[Double](1, 5, 5).apply1(e => Random.nextDouble())
    val output = model.updateOutput(input).toTensor[Double]

    val gradOutput = Tensor[Double]().resizeAs(output).apply1(e => Random.nextDouble())

    val gradInput = model.backward(input, gradOutput)

    val code = "torch.manualSeed(" + seed + ")\n" +
      """layer = nn.SpatialDivisiveNormalization()
      model = nn.Sequential()
      model:add(layer)
      model:zeroGradParameters()
      output = model:forward(input)
      gradInput = model:backward(input, gradOutput)
      """

    val (luaTime, torchResult) = TH.run(code,
      Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput")
    )

    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    output should be(luaOutput)
    gradInput should be(luaGradInput)
  }

  "A SpatialDivisiveNormalization" should "generate correct output and grad" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)

    val layer = new SpatialDivisiveNormalization[Double]()
    val model = new Sequential[Double]()
    model.add(layer)

    Random.setSeed(3)
    val input = Tensor[Double](3, 1, 5, 5).apply1(e => Random.nextDouble())
    val output = model.updateOutput(input).toTensor[Double]

    val gradOutput = Tensor[Double]().resizeAs(output).apply1(e => Random.nextDouble())

    val gradInput = model.backward(input, gradOutput)

    val code = "torch.manualSeed(" + seed + ")\n" +
      """layer = nn.SpatialDivisiveNormalization()
      model = nn.Sequential()
      model:add(layer)
      model:zeroGradParameters()
      output = model:forward(input)
      gradInput = model:backward(input, gradOutput)
      """

    val (luaTime, torchResult) = TH.run(code,
      Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput")
    )

    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    output should be(luaOutput)
    gradInput should be(luaGradInput)
  }

  "A SpatialDivisiveNormalization(4)" should "generate correct output and grad" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)

    val layer = new SpatialDivisiveNormalization[Double](4)
    val model = new Sequential[Double]()
    model.add(layer)

    Random.setSeed(3)
    val input = Tensor[Double](3, 4, 5, 5).apply1(e => Random.nextDouble())
    val output = model.updateOutput(input).toTensor[Double]

    val gradOutput = Tensor[Double]().resizeAs(output).apply1(e => Random.nextDouble())

    val gradInput = model.backward(input, gradOutput)

    val code = "torch.manualSeed(" + seed + ")\n" +
      """layer = nn.SpatialDivisiveNormalization(4)
      model = nn.Sequential()
      model:add(layer)
      model:zeroGradParameters()
      output = model:forward(input)
      gradInput = model:backward(input, gradOutput)
      """

    val (luaTime, torchResult) = TH.run(code,
      Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput")
    )

    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    output should be(luaOutput)
    gradInput should be(luaGradInput)
  }

  "SpatialDivisiveNormalization(4, kernel)" should "generate correct output and grad" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)

    val kernel = Tensor[Double](7).apply1(e => Random.nextDouble())

    val model = new SpatialDivisiveNormalization[Double](4, kernel)

    Random.setSeed(3)
    val input = Tensor[Double](3, 4, 5, 5).apply1(e => Random.nextDouble())
    val output = model.updateOutput(input)

    val gradOutput = Tensor[Double]().resizeAs(output).apply1(e => Random.nextDouble())

    val gradInput = model.backward(input, gradOutput)

    val code = "torch.manualSeed(" + seed + ")\n" +
      """model = nn.SpatialDivisiveNormalization(4, kernel)
      model:zeroGradParameters()
      output = model:forward(input)
      gradInput = model:backward(input, gradOutput)
      """

    val (luaTime, torchResult) = TH.run(code,
      Map("input" -> input, "gradOutput" -> gradOutput, "kernel" -> kernel),
      Array("output", "gradInput")
    )

    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    output should be(luaOutput)
    gradInput should be(luaGradInput)
  }
}
