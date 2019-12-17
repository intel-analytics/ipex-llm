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

import com.intel.analytics.bigdl.nn.{GradientChecker, SpatialAveragePooling}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._

import scala.math._
import scala.util.Random
import com.intel.analytics.bigdl._

@com.intel.analytics.bigdl.tags.Serial
class SpatialAveragePoolingSpec extends TorchSpec {
    "A SpatialAveragePooling" should "generate correct output and gradInput" in {
    torchCheck()
    val module = new SpatialAveragePooling[Double](3, 2, 2, 1)
    val input = Tensor[Double](1, 4, 3)
    input(Array(1, 1, 1)) = 0.25434372201562
    input(Array(1, 1, 2)) = 0.20443214406259
    input(Array(1, 1, 3)) = 0.33442943682894
    input(Array(1, 2, 1)) = 0.051310112234205
    input(Array(1, 2, 2)) = 0.56103343307041
    input(Array(1, 2, 3)) = 0.041837680386379
    input(Array(1, 3, 1)) = 0.75616162386723
    input(Array(1, 3, 2)) = 0.35945181339048
    input(Array(1, 3, 3)) = 0.4502888196148
    input(Array(1, 4, 1)) = 0.14862711215392
    input(Array(1, 4, 2)) = 0.050680571002886
    input(Array(1, 4, 3)) = 0.93014938035049
    val gradOutput = Tensor[Double](1, 3, 1)
    gradOutput(Array(1, 1, 1)) = 0.22147525195032
    gradOutput(Array(1, 2, 1)) = 0.30394183006138
    gradOutput(Array(1, 3, 1)) = 0.77438542619348

    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "module = nn.SpatialAveragePooling(3,2,2,1)\n" +
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

    println("Test case : SpatialAveragePooling, Torch : " + luaTime + " s, Scala : " +
      scalaTime / 1e9 + " s")
  }

  "A SpatialAveragePooling" should "be good in gradient checker for input" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)
    val layer = new SpatialAveragePooling[Double](3, 2, 2, 1)
    val input = Tensor[Double](1, 4, 3).apply1(e => Random.nextDouble())

    val checker = new GradientChecker(1e-4)
    checker.checkLayer[Double](layer, input, 1e-3) should be(true)
  }

}
