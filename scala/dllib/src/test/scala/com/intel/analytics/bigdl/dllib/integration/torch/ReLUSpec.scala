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

import com.intel.analytics.bigdl.nn.{GradientChecker, ReLU}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._

import scala.math._
import scala.util.Random
import com.intel.analytics.bigdl._

@com.intel.analytics.bigdl.tags.Serial
class ReLUSpec extends TorchSpec {
    "A ReLU Module " should "generate correct output and grad" in {
    torchCheck()
    val module = new ReLU[Double]()
    val input = Tensor[Double](2, 2, 2)
    input(Array(1, 1, 1)) = -0.97008799016476
    input(Array(1, 1, 2)) = -0.89318234380335
    input(Array(1, 2, 1)) = -0.65073125436902
    input(Array(1, 2, 2)) = -0.35406025126576
    input(Array(2, 1, 1)) = -1.0360766677186
    input(Array(2, 1, 2)) = 1.173689913936
    input(Array(2, 2, 1)) = 1.6776262558997
    input(Array(2, 2, 2)) = -0.64814318157732
    val gradOutput = Tensor[Double](2, 2, 2)
    gradOutput(Array(1, 1, 1)) = 0.43442418193445
    gradOutput(Array(1, 1, 2)) = 0.97614445211366
    gradOutput(Array(1, 2, 1)) = 0.081252868985757
    gradOutput(Array(1, 2, 2)) = 0.24688877537847
    gradOutput(Array(2, 1, 1)) = 0.027903598966077
    gradOutput(Array(2, 1, 2)) = 0.0086153273005038
    gradOutput(Array(2, 2, 1)) = 0.053113180678338
    gradOutput(Array(2, 2, 2)) = 0.74842141871341

    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "module = nn.ReLU()\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))
    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    luaOutput1.map(output, (v1, v2) => {
      assert(abs(v1 - v2) == 0);
      v1
    })
    luaOutput2.map(gradInput, (v1, v2) => {
      assert(abs(v1 - v2) == 0);
      v1
    })

    println("Test case : ReLU, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "ReLU module" should "be good in gradient check for input" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)
    val layer = new ReLU[Double]()
    val input = Tensor[Double](2, 2, 2).apply1(e => Random.nextDouble())

    val checker = new GradientChecker(1e-4)
    checker.checkLayer[Double](layer, input, 1e-3) should be(true)
  }
}
