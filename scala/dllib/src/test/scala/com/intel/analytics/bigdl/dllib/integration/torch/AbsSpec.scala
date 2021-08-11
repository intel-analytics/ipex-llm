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

import com.intel.analytics.bigdl.nn.Abs
import com.intel.analytics.bigdl.tensor.Tensor

@com.intel.analytics.bigdl.tags.Serial
class AbsSpec extends TorchSpec {
    "A Abs Module " should "generate correct output and grad" in {
    torchCheck()
    val module = new Abs[Double]
    val input = Tensor[Double](2, 1, 2)
    input(Array(1, 1, 1)) = 21
    input(Array(1, 1, 2)) = -29
    input(Array(2, 1, 1)) = -13
    input(Array(2, 1, 2)) = 27

    val gradOutput = Tensor[Double](2, 1, 2)
    gradOutput(Array(1, 1, 1)) = 10
    gradOutput(Array(1, 1, 2)) = -23
    gradOutput(Array(2, 1, 1)) = -10
    gradOutput(Array(2, 1, 2)) = 23

    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "module = nn.Abs()\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))

    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    luaOutput1.map(output, (v1, v2) => {
      assert(Math.abs(v1 - v2) == 0);
      v1
    })
    luaOutput2.map(gradInput, (v1, v2) => {
      assert(Math.abs(v1 - v2) == 0);
      v1
    })

    println("Test case : ReLU, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")

  }
}
