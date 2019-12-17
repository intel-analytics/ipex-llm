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

import com.intel.analytics.bigdl.nn.SoftMarginCriterion
import com.intel.analytics.bigdl.tensor.Tensor

import scala.util.Random

@com.intel.analytics.bigdl.tags.Serial
class SoftMarginCriterionSpec extends TorchSpec {
    "A SoftMarginCriterion Module " should "generate correct output and grad" in {
    torchCheck()
    val module = new SoftMarginCriterion[Double]()
    Random.setSeed(100)
    val input = Tensor[Double](4, 10).apply1(e => Random.nextDouble())
    val target = Tensor[Double](4, 10).randn()

    val start = System.nanoTime()
    val output = module.forward(input, target)
    val gradInput = module.backward(input, target)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "module = nn.SoftMarginCriterion()\n" +
      "output = module:forward(input, target)\n " +
      "gradInput = module:backward(input, target)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "target" -> target),
      Array("output", "gradInput"))
    val luaOutput = torchResult("output").asInstanceOf[Double]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    luaOutput should be(output)
    luaGradInput should be(gradInput)

    println("Test case : SoftMarginCriterion, Torch : " + luaTime + " s, Scala : "
      + scalaTime / 1e9 + " s")
  }

  "A SoftMarginCriterion Module with setting sizeAverage to false" should "generate " +
    "correct output and grad" in {
    torchCheck()
    val module = new SoftMarginCriterion[Double](false)
    Random.setSeed(100)
    val input = Tensor[Double](4, 10).apply1(e => Random.nextDouble())
    val target = Tensor[Double](4, 10).randn()

    val start = System.nanoTime()
    val output = module.forward(input, target)
    val gradInput = module.backward(input, target)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "module = nn.SoftMarginCriterion()\n" +
      "module.sizeAverage = false\n " +
      "output = module:forward(input, target)\n " +
      "gradInput = module:backward(input, target)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "target" -> target),
      Array("output", "gradInput"))
    val luaOutput = torchResult("output").asInstanceOf[Double]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    luaOutput should be(output)
    luaGradInput should be(gradInput)

    println("Test case : SoftMarginCriterion, Torch : " + luaTime + " s, Scala : "
      + scalaTime / 1e9 + " s")
  }
}
