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

import com.intel.analytics.bigdl.nn.Sigmoid
import com.intel.analytics.bigdl.tensor.Tensor

import scala.math._

@com.intel.analytics.bigdl.tags.Serial
class SigmoidSpec extends TorchSpec {
    "A Sigmoid Module " should "generate correct output and grad" in {
    torchCheck()
    val module = new Sigmoid[Double]
    val input = Tensor[Double](2, 2, 2)
    input(Array(1, 1, 1)) = 0.063364277360961
    input(Array(1, 1, 2)) = 0.90631252736785
    input(Array(1, 2, 1)) = 0.22275671223179
    input(Array(1, 2, 2)) = 0.37516756891273
    input(Array(2, 1, 1)) = 0.99284988618456
    input(Array(2, 1, 2)) = 0.97488326719031
    input(Array(2, 2, 1)) = 0.94414822547697
    input(Array(2, 2, 2)) = 0.68123375508003
    val gradOutput = Tensor[Double](2, 2, 2)
    gradOutput(Array(1, 1, 1)) = 0.38652365817688
    gradOutput(Array(1, 1, 2)) = 0.034144022269174
    gradOutput(Array(1, 2, 1)) = 0.68105488433503
    gradOutput(Array(1, 2, 2)) = 0.41517980070785
    gradOutput(Array(2, 1, 1)) = 0.91740695876069
    gradOutput(Array(2, 1, 2)) = 0.35317355184816
    gradOutput(Array(2, 2, 1)) = 0.24361599306576
    gradOutput(Array(2, 2, 2)) = 0.65869987895712

    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "module = nn.Sigmoid()\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))
    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    luaOutput1.map(output, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
    luaOutput2.map(gradInput, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })

    println("Test case : Sigmoid, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

}
