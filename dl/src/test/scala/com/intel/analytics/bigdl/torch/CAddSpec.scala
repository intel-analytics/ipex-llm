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

package com.intel.analytics.bigdl.torch

import com.intel.analytics.bigdl.nn.CAdd
import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}
import com.intel.analytics.bigdl.utils.RandomGenerator._

@com.intel.analytics.bigdl.tags.Serial
class CAddSpec extends FlatSpec with BeforeAndAfter with Matchers {
  before {
    if (!TH.hasTorch()) {
      cancel("Torch is not installed")
    }
  }

  "A CAdd(5, 1)" should "generate correct output and grad" in {
    val seed = 100
    RNG.setSeed(seed)

    val layer = new CAdd[Double](Array(5, 1))
    val input = Tensor[Double](5, 4)
    var i = 0
    input.apply1(_ => {i += 1; i})
    val gradOutput = Tensor[Double](5, 4)
    i = 0
    gradOutput.apply1(_ => {i += 1; i*0.1})

    val start = System.nanoTime()
    val output = layer.forward(input)
    val gradInput = layer.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "torch.manualSeed(" + seed + ")\n" +
      "module = nn.CAdd(5, 1)\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)\n" +
      "gradBias = module.gradBias"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput", "gradBias"))
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]
    val luaGradBias = torchResult("gradBias").asInstanceOf[Tensor[Double]]

    output should be (luaOutput)
    gradInput should be (luaGradInput)
    layer.gradBias should be (luaGradBias)

    println("Test case : CAdd, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "A CAdd(3)" should "generate correct output and grad" in {
    val seed = 100
    RNG.setSeed(seed)

    val layer = new CAdd[Double](Array(3))
    val input = Tensor[Double](2, 3)
    var i = 0
    input.apply1(_ => {i += 1; i})
    val gradOutput = Tensor[Double](2, 3)
    i = 0
    gradOutput.apply1(_ => {i += 1; i*0.1})

    val start = System.nanoTime()
    val output = layer.forward(input)
    val gradInput = layer.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "torch.manualSeed(" + seed + ")\n" +
      "module = nn.CAdd(3)\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)" +
      "gradBias = module.gradBias"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput", "gradBias"))
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]
    val luaGradBias = torchResult("gradBias").asInstanceOf[Tensor[Double]]

    output should be (luaOutput)
    gradInput should be (luaGradInput)
    layer.gradBias should be (luaGradBias)

    println("Test case : CAdd, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "A CAdd(3, 4)" should "generate correct output and grad" in {
    val seed = 100
    RNG.setSeed(seed)

    val layer = new CAdd[Double](Array(3, 4))
    val input = Tensor[Double](2, 3, 4)
    var i = 0
    input.apply1(_ => {i += 1; i})
    val gradOutput = Tensor[Double](2, 3, 4)
    i = 0
    gradOutput.apply1(_ => {i += 1; i*0.1})

    val start = System.nanoTime()
    val output = layer.forward(input)
    val gradInput = layer.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "torch.manualSeed(" + seed + ")\n" +
      "module = nn.CAdd(3, 4)\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)" +
      "gradBias = module.gradBias"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput", "gradBias"))
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]
    val luaGradBias = torchResult("gradBias").asInstanceOf[Tensor[Double]]

    output should be (luaOutput)
    gradInput should be (luaGradInput)
    layer.gradBias should be (luaGradBias)

    println("Test case : CAdd, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "A CAdd(1, 10, 1, 1)" should "generate correct output and grad" in {
    val seed = 100
    RNG.setSeed(seed)

    val layer = new CAdd[Double](Array(1, 10, 1, 1))
    val input = Tensor[Double](3, 10, 500, 600)
    var i = 0
    input.apply1(_ => {i += 1; i})
    val gradOutput = Tensor[Double](3, 10, 500, 600)
    i = 0
    gradOutput.apply1(_ => {i += 1; i*0.1})

    val start = System.nanoTime()
    val output = layer.forward(input)
    val gradInput = layer.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "torch.manualSeed(" + seed + ")\n" +
      "module = nn.CAdd(1, 10, 1, 1)\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)" +
      "gradBias = module.gradBias"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput", "gradBias"))
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]
    val luaGradBias = torchResult("gradBias").asInstanceOf[Tensor[Double]]

    output should be (luaOutput)
    gradInput should be (luaGradInput)
    layer.gradBias should be (luaGradBias)

    println("Test case : CAdd, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

}
