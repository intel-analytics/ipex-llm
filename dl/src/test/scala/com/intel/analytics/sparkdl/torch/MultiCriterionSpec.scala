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

import com.intel.analytics.sparkdl.nn._
import com.intel.analytics.sparkdl.tensor.{Storage, Tensor}
import com.intel.analytics.sparkdl.utils.Table
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.util.Random

class MultiCriterionSpec extends FlatSpec with BeforeAndAfter with Matchers{
  before {
    if (!TH.hasTorch()) {
      cancel("Torch is not installed")
    }
  }

  "A MultiCriterion Module " should "generate correct output and grad with Tensor input" in {
    val module = new MultiCriterion[Tensor[Double], Double]()
    val nll = new ClassNLLCriterion[Double]()
    val nll2 = new MSECriterion[Double]()
    module.add(nll.asInstanceOf[TensorCriterion[Double]], 0.5)
    module.add(nll2.asInstanceOf[TensorCriterion[Double]])

    val input = Tensor[Double](5).apply1(e => Random.nextDouble())
    val target = Tensor[Double](5)
    target(Array(1)) = 1
    target(Array(2)) = 2
    target(Array(3)) = 3
    target(Array(4)) = 2
    target(Array(5)) = 1

    val code = "nll = nn.ClassNLLCriterion()\n" +
      "nll2 = nn.MSECriterion()\n" +
      "module = nn.MultiCriterion():add(nll, 0.5):add(nll2)\n" +
      "output = module:forward(input, target)\n" +
      "gradInput = module:backward(input, target)\n"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "target" -> target),
      Array("output", "gradInput"))

    val luaOutput1 = torchResult("output").asInstanceOf[Double]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    val start = System.nanoTime()
    val output = module.forward(input, target)
    val gradInput = module.backward(input, target)
    val end = System.nanoTime()
    val scalaTime = end - start

    luaOutput1 should be(output)
    luaOutput2 should be(gradInput)

    println("Test case : MultiCriterion, Torch : " + luaTime +
      " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "A MultiCriterion Module " should "generate correct output and grad with Table input" in {
    val module = new MultiCriterion[Table, Double]()
    val nll = new CosineEmbeddingCriterion[Double]()
    val nll2 = new L1HingeEmbeddingCriterion[Double]()
    module.add(nll, 0.5)
    module.add(nll2)
/*
    val input1 = Tensor[Double](5).apply1(e => Random.nextDouble())
    val input2 = Tensor[Double](5).apply1(e => Random.nextDouble())
    val input = new Table()
    input(1.0) = input1
    input(2.0) = input2

    val target1 = Tensor[Double](5)
    target1(Array(1)) = 1
    target1(Array(2)) = 2
    target1(Array(3)) = 3
    target1(Array(4)) = 2
    target1(Array(5)) = 1

    val target2 = Tensor[Double](5)
    target2(Array(1)) = 1
    target2(Array(2)) = 4
    target2(Array(3)) = 2
    target2(Array(4)) = 4
    target2(Array(5)) = 5

    val target = new Table()
    target(1.0) = target1
    target(2.0) = target2
*/
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

    val code = "nll = nn.CosineEmbeddingCriterion()\n" +
      "nll2 = nn.L1HingeEmbeddingCriterion()\n" +
      "module = nn.MultiCriterion():add(nll, 0.5):add(nll2)\n" +
      "output = module:forward(input, target)\n" +
      "gradInput = module:backward(input, target)\n"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "target" -> target),
      Array("output", "gradInput"))

    val luaOutput1 = torchResult("output").asInstanceOf[Double]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    luaOutput1 should be(output)
    luaOutput2 should be(gradInput)

    println("Test case : MultiCriterion, Torch : " + luaTime +
      " s, Scala : " + scalaTime / 1e9 + " s")
  }
}
