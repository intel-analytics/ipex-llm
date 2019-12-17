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

import com.intel.analytics.bigdl.nn.L1Cost
import com.intel.analytics.bigdl.tensor.Tensor

import scala.util.Random

@com.intel.analytics.bigdl.tags.Serial
class L1CostSpec extends TorchSpec {
    "A L1Cost" should "generate correct output and grad" in {
    torchCheck()
    val layer = new L1Cost[Double]()
    val input = Tensor[Double](2, 2).apply1(e => Random.nextDouble())
    val target = Tensor[Double](2, 2).apply1(e => Random.nextDouble())

    val start = System.nanoTime()
    val output = layer.forward(input, target)
    val gradInput = layer.backward(input, target)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "module = nn.L1Cost()\n" +
      "output = module:forward(input, target)\n" +
      "gradInput = module:backward(input, target)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "target" -> target),
      Array("output", "gradInput"))
    val luaOutput = torchResult("output").asInstanceOf[Double]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    output should be (luaOutput)
    gradInput should be (luaGradInput)

    println("Test case : L1Cost, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }
}
