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

import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, MSECriterion, ParallelCriterion}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.{Engine, T, Table}

import scala.util.Random

@com.intel.analytics.bigdl.tags.Serial
class ParallelCriterionSpec extends TorchSpec {
    "A ParallelCriterion " should "generate correct output and grad" in {
    torchCheck()
    val seed = 100
    Random.setSeed(seed)

    val pc = new ParallelCriterion[Double]()
    val input1 = Tensor[Double](2, 10).apply1(_ => Random.nextDouble())
    val input2 = Tensor[Double](2, 10).apply1(_ => Random.nextDouble())
    val input = T()
    input(1.0) = input1
    input(2.0) = input2
    val target1 = Tensor[Double](Storage(Array(2.0, 5.0)))
    val target2 = Tensor[Double](2, 10).apply1(_ => Random.nextDouble())
    val target = T()
    target(1.0) = target1
    target(2.0) = target2
    val nll = new ClassNLLCriterion[Double]()
    val mse = new MSECriterion[Double]()
    pc.add(nll, 0.3).add(mse, 0.2)
    val start = System.nanoTime()
    val loss = pc.forward(input, target)
    val gradOutput = pc.backward(input, target)
    val scalaTime = System.nanoTime() - start

    val code = """
      nll = nn.ClassNLLCriterion()
      mse = nn.MSECriterion()
      pc = nn.ParallelCriterion():add(nll, 0.3):add(mse, 0.2)
      loss = pc:forward(input, target)
      gradOutput = pc:backward(input, target)
      gradOutput1 = gradOutput[1]
      gradOutput2 = gradOutput[2]
      """.stripMargin

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "target" -> target),
      Array("loss", "gradOutput1", "gradOutput2"))
    val luaLoss = torchResult("loss").asInstanceOf[Double]
    val luaGradOutput1 = torchResult("gradOutput1").asInstanceOf[Tensor[Double]]
    val luaGradOutput2 = torchResult("gradOutput2").asInstanceOf[Tensor[Double]]
    val luaGradOutput = T(luaGradOutput1, luaGradOutput2)

    luaLoss should be (loss)
    luaGradOutput should be (gradOutput)

    println("Test case : ParallelCriterion, Torch : " + luaTime +
      " s, Scala : " + scalaTime / 1e9 + " s")
  }
}


