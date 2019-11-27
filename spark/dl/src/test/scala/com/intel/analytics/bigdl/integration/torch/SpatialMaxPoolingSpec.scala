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

import com.intel.analytics.bigdl.nn.{GradientChecker, SpatialMaxPooling}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._

import scala.math._
import scala.util.Random
import com.intel.analytics.bigdl._

@com.intel.analytics.bigdl.tags.Serial
class SpatialMaxPoolingSpec extends TorchSpec {
    "A SpatialMaxPooling" should "generate correct output and gradInput" in {
    torchCheck()
    val module = new SpatialMaxPooling[Double](2, 2)
    val input = Tensor[Double](1, 3, 3)
    input(Array(1, 1, 1)) = 0.53367262030952
    input(Array(1, 1, 2)) = 0.79637692729011
    input(Array(1, 1, 3)) = 0.56747663160786
    input(Array(1, 2, 1)) = 0.18039962812327
    input(Array(1, 2, 2)) = 0.24608615692705
    input(Array(1, 2, 3)) = 0.22956256521866
    input(Array(1, 3, 1)) = 0.30736334621906
    input(Array(1, 3, 2)) = 0.59734606579877
    input(Array(1, 3, 3)) = 0.42989541869611
    val gradOutput = Tensor[Double](1, 1, 1)
    gradOutput(Array(1, 1, 1)) = 0.023921491578221

    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)"

    val (luaTime, torchResult) = TH.run(code, Map("module" -> module, "input" -> input,
      "gradOutput" -> gradOutput), Array("output", "gradInput"))
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

    println("Test case : SpatialMaxPooling, Torch : " + luaTime + " s, Scala : " +
      scalaTime / 1e9 + " s")
  }

  "A SpatialMaxPooling" should "be good in gradient check for input" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)
    val layer = new SpatialMaxPooling[Double](2, 2)
    val input = Tensor[Double](1, 3, 3).apply1(e => Random.nextDouble())

    val checker = new GradientChecker(1e-4)
    checker.checkLayer[Double](layer, input, 1e-3) should be(true)
  }
}
