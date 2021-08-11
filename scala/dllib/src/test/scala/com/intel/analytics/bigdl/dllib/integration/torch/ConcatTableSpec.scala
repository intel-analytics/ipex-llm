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

import com.intel.analytics.bigdl.nn.{ConcatTable, Linear}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.RandomGenerator._

import scala.util.Random

@com.intel.analytics.bigdl.tags.Serial
class ConcatTableSpec extends TorchSpec {
    "ConcatTable forward tensor" should "return right output" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)

    val ctable = new ConcatTable[Double]()
    ctable.zeroGradParameters()
    ctable.add(new Linear(5, 2))
    ctable.add(new Linear(5, 3))
    val input = Tensor[Double](5).apply1(_ => Random.nextDouble())
    val gradOutput1 = Tensor[Double](2).apply1(_ => Random.nextDouble())
    val gradOutput2 = Tensor[Double](3).apply1(_ => Random.nextDouble())

    val output = ctable.forward(input)

    val gradOutput = T(gradOutput1, gradOutput2)
    val gradInput = ctable.backward(input, gradOutput)

    val code = "torch.manualSeed(" + seed + ")\n" +
      """module = nn.ConcatTable():add(nn.Linear(5, 2)):add(nn.Linear(5, 3))
        module:zeroGradParameters()
        gradOutput = {gradOutput1, gradOutput2}
        output = module:forward(input)
        gradInput = module:backward(input, gradOutput)
        output1 = output[1]
        output2 = output[2]
        parameters, gradParameters = module:getParameters()
      """

    val (luaTime, torchResult) = TH.run(code,
      Map("input" -> input, "gradOutput1" -> gradOutput1, "gradOutput2" -> gradOutput2),
      Array("output1", "output2", "gradInput", "gradParameters"))
    val luaOutput1 = torchResult("output1").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("output2").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]
    val luaGradParameters = torchResult("gradParameters").asInstanceOf[Tensor[Double]]
    val luaOutput = T(luaOutput1, luaOutput2)

    val gradParameters = ctable.getParameters()._2.asInstanceOf[Tensor[Double]]

    output should be (luaOutput)
    gradInput should be (luaGradInput)
    gradParameters should be (luaGradParameters)
  }

}
