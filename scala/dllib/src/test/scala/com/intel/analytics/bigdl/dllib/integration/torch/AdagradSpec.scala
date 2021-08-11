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

import com.intel.analytics.bigdl.optim.Adagrad
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{RandomGenerator, T}

@com.intel.analytics.bigdl.tags.Serial
class AdagradSpec extends TorchSpec {
    "Adagrad with weightDecay" should "works fine" in {
    torchCheck()
    RandomGenerator.RNG.setSeed(10)
    val grad = Tensor[Float](10).rand()
    val param = Tensor[Float](10).rand()
    val adagrad = new Adagrad[Float](0.1, 5e-7, 0.01)

    val config = T("learningRate" -> 0.1, "learningRateDecay" -> 5e-7,
      "weightDecay" -> 0.01)

    val code = "function feval(x)\n return 1, grad\n end\n" +
      "optim.adagrad(feval, param, config)\n" +
      "optim.adagrad(feval, param, config)\n" +
      "optim.adagrad(feval, param, config)\n" +
      "optim.adagrad(feval, param, config)\n" +
      "x,f = optim.adagrad(feval, param, config)\n"

    val (luaTime, torchResult) = TH.run(code, Map("grad" -> grad, "param" -> param,
      "config" -> config),
      Array("x", "grad"))
    val luaGrad = torchResult("grad").asInstanceOf[Tensor[Float]]
    val luaParam = torchResult("x").asInstanceOf[Tensor[Float]]

    for (i <- 1 to 5) {
      adagrad.optimize(_ => (1f, grad), param)
    }

    luaParam should be (param)
    luaGrad should be (grad)
  }
}
