/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
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

import com.intel.analytics.bigdl.nn.MultiLabelSoftMarginCriterion
import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

@com.intel.analytics.bigdl.tags.Parallel
class MultiLabelSoftMarginCriterionSpec extends FlatSpec with BeforeAndAfter with Matchers{
  before {
    if (!TH.hasTorch()) {
      cancel("Torch is not installed")
    }
  }

  "A MultiLabelSoftMarginCriterion Criterion " should
    "generate correct output and grad" in {
    val criterion = new MultiLabelSoftMarginCriterion[Double]()

    val input = Tensor[Double](3)
    input(Array(1)) = 0.4
    input(Array(2)) = 0.5
    input(Array(3)) = 0.6

    val target = Tensor[Double](3)
    target(Array(1)) = 0
    target(Array(2)) = 1
    target(Array(3)) = 1

    val start = System.nanoTime()
    val output = criterion.forward(input, target)
    val gradInput = criterion.backward(input, target)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "abs = nn.MultiLabelSoftMarginCriterion()\n" +
      "output = abs:forward(input, target)\n " +
      "gradInput = abs:backward(input, target)"


    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "target" -> target),
      Array("output", "gradInput"))
    val luaOutput1 = torchResult("output").asInstanceOf[Double]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    luaOutput1 should be(output)
    luaOutput2 should be(gradInput)

    println("Test case : MultiLabelSoftMarginCriterion, Torch : " +
      luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

}
