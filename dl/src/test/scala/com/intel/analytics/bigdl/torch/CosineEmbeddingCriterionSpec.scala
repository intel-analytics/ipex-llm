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
package com.intel.analytics.bigdl.torch

import com.intel.analytics.bigdl.nn.CosineEmbeddingCriterion
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.Table

import scala.collection.mutable.HashMap
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.util.Random

class CosineEmbeddingCriterionSpec extends FlatSpec with BeforeAndAfter with Matchers {
  before {
    if (!TH.hasTorch()) {
      cancel("Torch is not installed")
    }
  }

  "A CosineEmbeddingCriterion Module" should "generate correct output and grad" in {
    val seed = 100
    RNG.setSeed(seed)
    val module = new CosineEmbeddingCriterion[Double](0.2)

    val input1 = Tensor[Double](5).apply1(e => Random.nextDouble())
    val input2 = Tensor[Double](5).apply1(e => Random.nextDouble())
    val input = new Table()
    input(1.0) = input1
    input(2.0) = input2

    val target = new Table()
    val target1 = Tensor[Double](Storage(Array(-0.5)))
    target(1.toDouble) = target1

    val start = System.nanoTime()
    val output = module.forward(input, target)
    val gradInput = module.backward(input, target)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "torch.manualSeed(" + seed + ")\n" +
      "module = nn.CosineEmbeddingCriterion(0.2)\n" +
      "_idx = module._idx\n" +
      "_outputs = module._outputs\n" +
      "buffer = module.buffer\n" +
      "output = module:forward(input, -0.5)\n" +
      "gradInput = module:backward(input, -0.5)\n"


    val (luaTime, torchResult) = TH.run(code, Map("input" -> input),
      Array("output", "gradInput", "_idx", "buffer", "_outputs"))
    val luaOutput1 = torchResult("output").asInstanceOf[Double]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Table]

    luaOutput1 should be(output)
    luaOutput2 should be (gradInput)

    println("Test case : CrossEntropyCriterion, Torch : " + luaTime +
      " s, Scala : " + scalaTime / 1e9 + " s")
  }
}
