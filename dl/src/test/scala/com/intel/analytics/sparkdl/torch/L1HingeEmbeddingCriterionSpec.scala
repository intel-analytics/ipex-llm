/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.intel.analytics.sparkdl.torch

import com.intel.analytics.sparkdl.nn.{L1HingeEmbeddingCriterion}
import com.intel.analytics.sparkdl.tensor.{Storage, Tensor}
import com.intel.analytics.sparkdl.utils.RandomGenerator._
import com.intel.analytics.sparkdl.utils.Table
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.collection.mutable.HashMap


class L1HingeEmbeddingCriterionSpec extends FlatSpec with BeforeAndAfter with Matchers{
  before {
    if (!TH.hasTorch()) {
      cancel("Torch is not installed")
    }
  }

  "A L1HingeEmbeddingCriterion" should "generate correct output and grad with different " +
    "input tensor" in {
    val seed = 100
    RNG.setSeed(seed)
    val module = new L1HingeEmbeddingCriterion[Double](0.6)

    val input1 = Tensor[Double](Storage(Array(1.0, 0.0)))
    val input2 = Tensor[Double](Storage(Array(0.5, math.sqrt(3) * 0.5)))
    var input = new Table()
    input(1.toDouble) = input1
    input(2.toDouble) = input2

    val target = new Table()
    val target1 = Tensor[Double](Storage(Array(-1.0)))
    target(1.toDouble) = target1

    val start = System.nanoTime()
    val output = module.forward(input, target)
    val gradInput = module.backward(input, target)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "torch.manualSeed(" + seed + ")\n" +
      "module = nn.L1HingeEmbeddingCriterion(0.6)\n" +
      "output = module:forward(input, -1)\n" +
      "gradInput = module:backward(input, -1)\n"


    val (luaTime, torchResult) = TH.run(code, Map("input" -> input), Array("output", "gradInput"))
    val luaOutput1 = torchResult("output").asInstanceOf[Double]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[HashMap[Double, Tensor[Double]]]

    luaOutput1 should be(output)
    val luagradInput1 = luaOutput2.get(1.0).getOrElse(null)
    val luagradInput2 = luaOutput2.get(2.0).getOrElse(null)

    val gradInput1 = gradInput.apply(1.toDouble).asInstanceOf[Tensor[Double]]
    val gradInput2 = gradInput.apply(2.toDouble).asInstanceOf[Tensor[Double]]
    gradInput1 should be(luagradInput1)
    gradInput2 should be(luagradInput2)

    println("Test case : L1HingeEmbeddingCriterion, Torch : " + luaTime +
      " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "A L1HingeEmbeddingCriterion" should "generate correct output and grad with same " +
    "input tensor" in {
    val seed = 100
    RNG.setSeed(seed)
    val module = new L1HingeEmbeddingCriterion[Double](0.6)

    val input1 = Tensor[Double](Storage(Array(1.0, 0.8)))
    val input2 = Tensor[Double](Storage(Array(1.0, 0.6)))
    var input = new Table()
    input(1.toDouble) = input1
    input(2.toDouble) = input2

    val target = new Table()
    val target1 = Tensor[Double](Storage(Array(-1.0)))
    target(1.toDouble) = target1

    val start = System.nanoTime()
    val output = module.forward(input, target)
    val gradInput = module.backward(input, target)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "torch.manualSeed(" + seed + ")\n" +
      "module = nn.L1HingeEmbeddingCriterion(0.6)\n" +
      "output = module:forward(input, -1)\n" +
      "gradInput = module:backward(input, -1)\n"


    val (luaTime, torchResult) = TH.run(code, Map("input" -> input), Array("output", "gradInput"))
    val luaOutput1 = torchResult("output").asInstanceOf[Double]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[HashMap[Double, Tensor[Double]]]

    luaOutput1 should be(output)
    val luagradInput1 = luaOutput2.get(1.0).getOrElse(null)
    val luagradInput2 = luaOutput2.get(2.0).getOrElse(null)

    val gradInput1 = gradInput.apply(1.toDouble).asInstanceOf[Tensor[Double]]
    val gradInput2 = gradInput.apply(2.toDouble).asInstanceOf[Tensor[Double]]
    gradInput1 should be(luagradInput1)
    gradInput2 should be(luagradInput2)

    println("Test case : L1HingeEmbeddingCriterion, Torch : " + luaTime +
      " s, Scala : " + scalaTime / 1e9 + " s")
  }
}
