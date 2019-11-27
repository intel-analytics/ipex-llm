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

import com.intel.analytics.bigdl.nn.Normalize
import com.intel.analytics.bigdl.tensor.Tensor

import scala.util.Random

@com.intel.analytics.bigdl.tags.Serial
class NormalizeSpec extends TorchSpec {
    "A Normalize Module" should "generate correct output and grad with input one dimension" in {
    torchCheck()
    val p = 1.5
    val input = Tensor[Double](9).apply1(e => Random.nextDouble())
    val gradOutput = Tensor[Double](9).apply1(e => Random.nextDouble())

    val code = "module = nn.Normalize(" + p + ")\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))

    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    val module = Normalize[Double](p)
    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    output should be(luaOutput1)
    gradInput should be(luaOutput2)

    println("Test case : Normalize, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "A Normalize Module" should "generate correct output and grad with input two dimensions" in {
    torchCheck()
    val input = Tensor[Double](2, 9).apply1(e => Random.nextDouble())
    val gradOutput = Tensor[Double](2, 9).apply1(e => Random.nextDouble())

    val code = "module = nn.Normalize(math.huge)\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))

    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    val module = Normalize[Double](Double.MaxValue)
    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    output should be(luaOutput1)
    gradInput should be(luaOutput2)

    println("Test case : Normalize, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "A Normalize Module multiple time" should "generate correct" +
    " output and grad with input two dimensions" in {
    torchCheck()
    val input = Tensor[Double](2, 9).apply1(e => Random.nextDouble())
    val gradOutput = Tensor[Double](2, 9).apply1(e => Random.nextDouble())

    val code = "module = nn.Normalize(math.huge)\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)"


    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))

    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    val module = Normalize[Double](Double.MaxValue)
    val start = System.nanoTime()
    var output = module.forward(input)
    var gradInput = module.backward(input, gradOutput)
    output = module.forward(input)
    gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    output should be(luaOutput1)
    gradInput should be(luaOutput2)

    println("Test case : Normalize, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }
}
