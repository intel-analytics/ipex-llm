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

import com.intel.analytics.bigdl.nn.CosineDistanceCriterion
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.{RandomGenerator, Table}

import scala.collection.mutable.HashMap

import scala.util.Random

@com.intel.analytics.bigdl.tags.Serial
class CosineDistanceCriterionSpec extends TorchSpec {
  "A CosineDistanceCriterionSpec Module" should "generate correct output and grad" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)
    val module = CosineDistanceCriterion[Double](false)

    val input1 = Tensor[Double](5).apply1(e => RandomGenerator.RNG.uniform(0, 2))
    val input2 = Tensor[Double](5).apply1(e => RandomGenerator.RNG.uniform(0, 1))
    val input = new Table()
    input(1.0) = input1
    input(2.0) = input2

    val target = new Table()
    val target1 = Tensor[Double](Storage(Array(1.0)))
    target(1.toDouble) = target1

    val start = System.nanoTime()
    val output = module.forward(input1, input2)
    val gradInput = module.backward(input1, input2)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "torch.manualSeed(" + seed + ")\n" +
      "module = nn.CosineEmbeddingCriterion(0.0)\n" +
      "_idx = module._idx\n" +
      "_outputs = module._outputs\n" +
      "buffer = module.buffer\n" +
      "output = module:forward(input, 1.0)\n" +
      "gradInput = module:backward(input, 1.0)\n"


    val (luaTime, torchResult) = TH.run(code, Map("input" -> input),
      Array("output", "gradInput", "_idx", "buffer", "_outputs"))
    val luaOutput1 = torchResult("output").asInstanceOf[Double]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Table]

    luaOutput1 should be(output)
    luaOutput2[Tensor[Double]](1) should be (gradInput.squeeze())

    println("Test case : CrossEntropyCriterion, Torch : " + luaTime +
      " s, Scala : " + scalaTime / 1e9 + " s")
  }
}
