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

import com.intel.analytics.bigdl.nn.Padding
import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.util.Random

@com.intel.analytics.bigdl.tags.Serial
class PaddingSpec extends TorchSpec {


  "A Padding Module " should "generate correct output and grad with nInputDim != input.dim()" in {
    torchCheck()
    val dim = 1
    val pad = -1
    val nInputDim = 4
    val value = -0.8999761
    val index = 14

    val input = Tensor[Double](3, 13, 11).apply1(e => Random.nextDouble())
    val gradOutput = Tensor[Double](3, 14, 11).apply1(e => Random.nextDouble())

    val code = "module = nn.Padding(" + dim + "," + pad + "," + nInputDim + "," +
      value + "," + index + ")\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)\n"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))

    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    val module = Padding[Double](dim, pad, nInputDim, value, index)
    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    output should be(luaOutput1)
    gradInput should be(luaOutput2)

    println("Test case : Padding, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "A Padding Module " should "generate correct output and grad with nInputDim == input.dim()" in {
    torchCheck()
    val dim = 1
    val pad = 1
    val nInputDim = 3
    val value = 1
    val index = 2

    val input = Tensor[Double](3, 13, 11).apply1(e => Random.nextDouble())
    val gradOutput = Tensor[Double](5, 13, 11).apply1(e => Random.nextDouble())

    val code = "module = nn.Padding(" + dim + "," + pad + "," + nInputDim + "," +
      value + "," + index + ")\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)\n"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))

    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    val module = Padding[Double](dim, pad, nInputDim, value, index)
    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    output should be(luaOutput1)
    gradInput should be(luaOutput2)

    println("Test case : Padding, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "A Padding Module " should "generate correct output and grad with index == 1" in {
    torchCheck()
    val dim = 1
    val pad = -1
    val nInputDim = 4
    val value = -0.8999761
    val index = 1

    val input = Tensor[Double](3, 13, 11).apply1(e => Random.nextDouble())
    val gradOutput = Tensor[Double](3, 14, 11).apply1(e => Random.nextDouble())

    val code = "module = nn.Padding(" + dim + "," + pad + "," + nInputDim + "," +
      value + "," + index + ")\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)\n"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))

    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    val module = Padding[Double](dim, pad, nInputDim, value, index)
    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    output should be(luaOutput1)
    gradInput should be(luaOutput2)

    println("Test case : Padding, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }
}

