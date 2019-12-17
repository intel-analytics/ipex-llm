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
import com.intel.analytics.bigdl.nn.MarginRankingCriterion
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.Table

import scala.collection.mutable.HashMap
import scala.util.Random

@com.intel.analytics.bigdl.tags.Serial
class MarginRankingCriterionSpec extends TorchSpec {
    "A MarginRankingCriterion " should "generate correct output and grad with only value" in {
    torchCheck()
    val mse = new MarginRankingCriterion[Double]()

    val input1 = Tensor[Double](5).apply1(e => Random.nextDouble())
    val input2 = Tensor[Double](5).apply1(e => Random.nextDouble())

    val input = new Table()
    input(1.toDouble) = input1
    input(2.toDouble) = input2

    val target = new Table()
    val target1 = Tensor[Double](Storage(Array(-1.0)))
    target(1.toDouble) = target1

    val start = System.nanoTime()
    val output = mse.forward(input, target)
    val gradInput = mse.backward(input, target)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "mse = nn.MarginRankingCriterion()\n" +
      "output = mse:forward(input,-1)\n" +
      "gradInput = mse:backward(input,-1)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "target" -> target),
      Array("output", "gradInput"))
    val luaOutput1 = torchResult("output").asInstanceOf[Double]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Table]

    luaOutput1 should be (output)
    gradInput should equal (luaOutput2)

    println("Test case : MarginRankingCriterion, Torch : " + luaTime +
      " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "A MarginRankingCriterion " should "generate correct output and grad with Tensor target" in {
    torchCheck()
    val mse = new MarginRankingCriterion[Double]()

    val input1 = Tensor[Double](5).apply1(e => Random.nextDouble())
    val input2 = Tensor[Double](5).apply1(e => Random.nextDouble())

    val input = new Table()
    input(1.toDouble) = input1
    input(2.toDouble) = input2

    val target = new Table()
    val target1 = Tensor[Double](5).apply1(e => Random.nextDouble())
    target(1.toDouble) = target1

    val start = System.nanoTime()
    val output = mse.forward(input, target)
    val gradInput = mse.backward(input, target)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "mse = nn.MarginRankingCriterion()\n" +
      "output = mse:forward(input, target)\n" +
      "gradInput = mse:backward(input, target)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "target" -> target1),
      Array("output", "gradInput"))
    val luaOutput1 = torchResult("output").asInstanceOf[Double]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Table]

    luaOutput1 should be (output)
    gradInput should equal (luaOutput2)

    println("Test case : MarginRankingCriterion, Torch : " + luaTime +
      " s, Scala : " + scalaTime / 1e9 + " s")
  }
}
