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

import com.intel.analytics.bigdl.nn.HingeEmbeddingCriterion
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator

@com.intel.analytics.bigdl.tags.Serial
class HingeEmbeddingCriterionSpec extends TorchSpec {
    "A HingeEmbeddingCriterion" should "generate correct output and grad" in {
    torchCheck()
    val module = new HingeEmbeddingCriterion[Double](0.6)

    val input = Tensor[Double](10).apply1(x => RandomGenerator.RNG.uniform(0, 2))

    val target = Tensor[Double](10).apply1(x =>
      if (RandomGenerator.RNG.uniform(-1, 1) > 0) {
        1.0
      } else {
        -1.0
    })

    val start = System.nanoTime()
    val output = module.forward(input, target)
    val gradInput = module.backward(input, target)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "module = nn.HingeEmbeddingCriterion(0.6)\n" +
      "output = module:forward(input, target)\n" +
      "gradInput = module:backward(input, target)\n"


    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "target" -> target),
      Array("output", "gradInput"))
    val luaOutput1 = torchResult("output").asInstanceOf[Double]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    luaOutput1 should be (output)
    luaOutput2 should be (gradInput)

    println("Test case : HingeEmbeddingCriterion, Torch : " + luaTime + " s, Scala : " +
      scalaTime / 1e9 + " s")
  }
}
