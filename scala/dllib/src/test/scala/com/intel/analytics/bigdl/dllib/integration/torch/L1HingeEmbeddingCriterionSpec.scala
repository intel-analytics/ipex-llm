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

import com.intel.analytics.bigdl.nn.L1HingeEmbeddingCriterion
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.Table

import scala.collection.mutable.HashMap
import scala.util.Random

@com.intel.analytics.bigdl.tags.Serial
class L1HingeEmbeddingCriterionSpec extends TorchSpec {
    "A L1HingeEmbeddingCriterion" should "generate correct output and grad with y == 1 " in {
    torchCheck()
    val seed = 2
    RNG.setSeed(seed)
    val module = new L1HingeEmbeddingCriterion[Double](0.6)

    val input1 = Tensor[Double](2).apply1(e => Random.nextDouble())
    val input2 = Tensor[Double](2).apply1(e => Random.nextDouble())
    val input = new Table()
    input(1.0) = input1
    input(2.0) = input2

    val target = Tensor[Double](1)
    target(Array(1)) = 1.0

    val start = System.nanoTime()
    val output = module.forward(input, target)
    val gradInput = module.backward(input, target)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "torch.manualSeed(" + seed + ")\n" +
      "module = nn.L1HingeEmbeddingCriterion(0.6)\n" +
      "output = module:forward(input, 1)\n" +
      "gradInput = module:backward(input, 1)\n"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input), Array("output", "gradInput"))
    val luaOutput1 = torchResult("output").asInstanceOf[Double]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Table]

    luaOutput1 should be(output)
    luaOutput2 should be (gradInput)

    println("Test case : L1HingeEmbeddingCriterion, Torch : " + luaTime +
      " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "A L1HingeEmbeddingCriterion" should "generate correct output and grad with y == -1 " in {
    torchCheck()
    val seed = 2
    RNG.setSeed(seed)
    val module = new L1HingeEmbeddingCriterion[Double](0.6)

    val input1 = Tensor[Double](2).apply1(e => Random.nextDouble())
    val input2 = Tensor[Double](2).apply1(e => Random.nextDouble())
    val input = new Table()
    input(1.0) = input1
    input(2.0) = input2

    val target = Tensor[Double](1)
    target(Array(1)) = -1.0

    val start = System.nanoTime()
    val output = module.forward(input, target)
    val gradInput = module.backward(input, target)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "torch.manualSeed(" + seed + ")\n" +
      "module = nn.L1HingeEmbeddingCriterion(0.6)\n" +
      "output = module:forward(input, -1.0)\n" +
      "gradInput = module:backward(input, -1.0)\n"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input), Array("output", "gradInput"))
    val luaOutput1 = torchResult("output").asInstanceOf[Double]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Table]

    luaOutput1 should be(output)
    luaOutput2 should be (gradInput)

    println("Test case : L1HingeEmbeddingCriterion, Torch : " + luaTime +
      " s, Scala : " + scalaTime / 1e9 + " s")
  }
}
