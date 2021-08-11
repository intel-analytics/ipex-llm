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

import com.intel.analytics.bigdl.nn.PairwiseDistance
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T

import scala.util.Random

@com.intel.analytics.bigdl.tags.Serial
class PairwiseDistanceSpec extends TorchSpec {
    "A PairwiseDistance with one dimension input" should "generate correct output and grad" in {
    torchCheck()
    val seed = 100
    Random.setSeed(seed)

    val module = new PairwiseDistance[Double](1)
    val input1 = Tensor[Double](10).apply1(_ => Random.nextDouble())
    val input2 = Tensor[Double](10).apply1(_ => Random.nextDouble())
    val input = T(1.0 -> input1, 2.0 -> input2)
    val gradOutput = Tensor[Double](1).randn()
    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val scalaTime = System.nanoTime() - start

    val code =
      s"""
      torch.manualSeed($seed)
      module = nn.PairwiseDistance(1)
      output = module:forward(input)
      gradInput = module:backward(input, gradOutput)
      gradInput1 = gradInput[1]
      gradInput2 = gradInput[2]
               """.stripMargin

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput1", "gradInput2"))
    val torchOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val torchgradInput1 = torchResult("gradInput1").asInstanceOf[Tensor[Double]]
    val torchgradInput2 = torchResult("gradInput2").asInstanceOf[Tensor[Double]]
    val torchgradInput = T(torchgradInput1, torchgradInput2)

    torchOutput should be (output)
    torchgradInput should be (gradInput)

    println("Test case : PairwiseDistance, Torch : " + luaTime +
      " s, Scala : " + scalaTime / 1e9 +
      " s")
  }

  "A PairwiseDistance with two dimension input" should "generate correct output and grad" in {
    torchCheck()
    val seed = 100
    Random.setSeed(seed)

    val module = new PairwiseDistance[Double](5)
    val input1 = Tensor[Double](5, 10).apply1(_ => Random.nextDouble())
    val input2 = Tensor[Double](5, 10).apply1(_ => Random.nextDouble())
    val input = T(1.0 -> input1, 2.0 -> input2)
    val gradOutput = Tensor[Double](5).randn()
    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val scalaTime = System.nanoTime() - start

    val code =
      s"""
      torch.manualSeed($seed)
      module = nn.PairwiseDistance(5)
      output = module:forward(input)
      gradInput = module:backward(input, gradOutput)
      gradInput1 = gradInput[1]
      gradInput2 = gradInput[2]
               """.stripMargin

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput1", "gradInput2"))
    val torchOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val torchgradInput1 = torchResult("gradInput1").asInstanceOf[Tensor[Double]]
    val torchgradInput2 = torchResult("gradInput2").asInstanceOf[Tensor[Double]]
    val torchgradInput = T(torchgradInput1, torchgradInput2)

    torchOutput should be (output)
    torchgradInput should be (gradInput)

    println("Test case : PairwiseDistance, Torch : " + luaTime +
      " s, Scala : " + scalaTime / 1e9 +
      " s")
  }
}
