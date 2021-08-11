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

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.{GradientChecker, TemporalMaxPooling}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._

import scala.math._
import scala.util.Random

class TemporalMaxPoolingSpec extends TorchSpec {
  "A TemporalMaxPooling" should "generate correct output and gradInput" in {
    torchCheck()
    RNG.setSeed(100)
    val module = new TemporalMaxPooling[Double](4)
    val input = Tensor[Double](5, 4, 5).rand()
    val gradOutput = Tensor[Double](5, 1, 5).rand()

    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code =
      s"""
         |module = nn.TemporalMaxPooling(4)
         |output = module:forward(input)
         |gradInput = module:backward(input,gradOutput)
       """.stripMargin

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input,
      "gradOutput" -> gradOutput), Array("output", "gradInput"))
    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Tensor[Double]]


    luaOutput1.map(output, (v1, v2) => {
      assert(abs(v1 - v2) == 0)
      v1
    })
    luaOutput2.map(gradInput, (v1, v2) => {
      assert(abs(v1 - v2) == 0)
      v1
    })

    println("Test case : TemporalMaxPooling, Torch : " + luaTime + " s, Scala : " +
      scalaTime / 1e9 + " s")
  }

  "A TemporalMaxPooling" should "generate correct output and gradInput in batch" in {
    torchCheck()
    RNG.setSeed(100)
    val module = new TemporalMaxPooling[Double](4)
    val input = Tensor[Double](5, 10, 5).rand()

    val start = System.nanoTime()
    val output = module.forward(input)
    val gradOutput = Tensor[Double](5, 2, 5).rand()
    val gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code =
      s"""
         |module = nn.TemporalMaxPooling(4)
         |output = module:forward(input)
         |gradInput = module:backward(input,gradOutput)
       """.stripMargin

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input,
      "gradOutput" -> gradOutput), Array("output", "gradInput"))
    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Tensor[Double]]


    luaOutput1.map(output, (v1, v2) => {
      assert(abs(v1 - v2) == 0)
      v1
    })
    luaOutput2.map(gradInput, (v1, v2) => {
      assert(abs(v1 - v2) == 0)
      v1
    })

    println("Test case : TemporalMaxPooling, Torch : " + luaTime + " s, Scala : " +
      scalaTime / 1e9 + " s")
  }

  "A TemporalMaxPooling" should "generate correct output and gradInput in batch with" +
    "multiple forward and backward" in {
    torchCheck()
    RNG.setSeed(100)
    val module = new TemporalMaxPooling[Double](4)
    val input = Tensor[Double](5, 10, 5).rand()
    val gradOutput = Tensor[Double](5, 2, 5).rand()

    val start = System.nanoTime()
    var output = module.forward(input)
    var gradInput = module.backward(input, gradOutput)
    output = module.forward(input)
    gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code =
      s"""
         |module = nn.TemporalMaxPooling(4)
         |output = module:forward(input)
         |gradInput = module:backward(input,gradOutput)
         |output = module:forward(input)
         |gradInput = module:backward(input,gradOutput)
       """.stripMargin

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input,
      "gradOutput" -> gradOutput), Array("output", "gradInput"))
    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Tensor[Double]]


    luaOutput1.map(output, (v1, v2) => {
      assert(abs(v1 - v2) == 0)
      v1
    })
    luaOutput2.map(gradInput, (v1, v2) => {
      assert(abs(v1 - v2) == 0)
      v1
    })

    println("Test case : TemporalMaxPooling, Torch : " + luaTime + " s, Scala : " +
      scalaTime / 1e9 + " s")
  }
  "A TemporalMaxPooling" should "be good in gradient check for input" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)
    val layer = new TemporalMaxPooling[Double](2, 2)
    val input = Tensor[Double](1, 3, 3).apply1(e => Random.nextDouble())

    val checker = new GradientChecker(1e-4)
    checker.checkLayer[Double](layer, input, 1e-3) should be(true)
  }
}
