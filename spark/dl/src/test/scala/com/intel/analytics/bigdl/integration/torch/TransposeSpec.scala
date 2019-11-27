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

import com.intel.analytics.bigdl.nn.Transpose
import com.intel.analytics.bigdl.tensor.Tensor

import scala.math._

class TransposeSpec extends TorchSpec {
    "A Transpose Module " should "generate correct output and grad" in {
    torchCheck()
    val module = new Transpose[Double](Array((1, 3)))
    val input = Tensor[Double](2, 2, 2)
    input(Array(1, 1, 1)) = -0.17020166106522
    input(Array(1, 1, 2)) = 0.57785657607019
    input(Array(1, 2, 1)) = -1.3404131438583
    input(Array(1, 2, 2)) = 1.0938102817163
    input(Array(2, 1, 1)) = 1.120370157063
    input(Array(2, 1, 2)) = -1.5014141565189
    input(Array(2, 2, 1)) = 0.3380249235779
    input(Array(2, 2, 2)) = -0.625677742064
    val gradOutput = Tensor[Double](2, 2, 2)
    gradOutput(Array(1, 1, 1)) = 0.79903302760795
    gradOutput(Array(1, 1, 2)) = 0.019753993256018
    gradOutput(Array(1, 2, 1)) = 0.63136631483212
    gradOutput(Array(1, 2, 2)) = 0.29849314852618
    gradOutput(Array(2, 1, 1)) = 0.94380705454387
    gradOutput(Array(2, 1, 2)) = 0.030344664584845
    gradOutput(Array(2, 2, 1)) = 0.33804601291195
    gradOutput(Array(2, 2, 2)) = 0.8807330634445

    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "module = nn.Transpose({1, 3})\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))
    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    luaOutput1.map(output.asInstanceOf[Tensor[Double]], (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
    luaOutput2.map(gradInput.asInstanceOf[Tensor[Double]], (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })

    println("Test case : Transpose, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "A Transpose Module with mutiple tuples" should "generate correct output and grad" in {
    torchCheck()
    val module = new Transpose[Double](Array((1, 3), (2, 3)))
    val input = Tensor[Double](2, 2, 2)
    input(Array(1, 1, 1)) = -0.17020166106522
    input(Array(1, 1, 2)) = 0.57785657607019
    input(Array(1, 2, 1)) = -1.3404131438583
    input(Array(1, 2, 2)) = 1.0938102817163
    input(Array(2, 1, 1)) = 1.120370157063
    input(Array(2, 1, 2)) = -1.5014141565189
    input(Array(2, 2, 1)) = 0.3380249235779
    input(Array(2, 2, 2)) = -0.625677742064
    val gradOutput = Tensor[Double](2, 2, 2)
    gradOutput(Array(1, 1, 1)) = 0.79903302760795
    gradOutput(Array(1, 1, 2)) = 0.019753993256018
    gradOutput(Array(1, 2, 1)) = 0.63136631483212
    gradOutput(Array(1, 2, 2)) = 0.29849314852618
    gradOutput(Array(2, 1, 1)) = 0.94380705454387
    gradOutput(Array(2, 1, 2)) = 0.030344664584845
    gradOutput(Array(2, 2, 1)) = 0.33804601291195
    gradOutput(Array(2, 2, 2)) = 0.8807330634445

    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "module = nn.Transpose({1, 3}, {2, 3})\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))
    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    luaOutput1.map(output.asInstanceOf[Tensor[Double]], (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
    luaOutput2.map(gradInput.asInstanceOf[Tensor[Double]], (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })

    println("Test case : Transpose, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }
}
