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

import com.intel.analytics.bigdl.nn.{Linear, Sequential}
import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.math._

@com.intel.analytics.bigdl.tags.Parallel
class ModuleSpec extends FlatSpec with BeforeAndAfter with Matchers {
  before {
    if (!TH.hasTorch()) {
      cancel("Torch is not installed")
    }
  }

  "getParameter" should "behave correctly" in {
    val module = new Sequential[Double]
    val subModule1 = new Linear[Double](2, 3)
    val subModule2 = new Linear[Double](4, 5)
    module.add(subModule1)
    module.add(subModule2)

    val (weight, grad) = module.getParameters()

    val code = "module = nn.Sequential()\n" +
      "module:add(subModule1)\n" +
      "module:add(subModule2)\n" +
      "weight, grad = module:getParameters()\n"


    val (luaTime, torchResult) = TH.run(code, Map("subModule1" -> subModule1,
      "subModule2" -> subModule2), Array("weight", "grad"))
    val luaOutput1 = torchResult("weight").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("grad").asInstanceOf[Tensor[Double]]

    luaOutput1.map(weight, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
    luaOutput2.map(grad, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })


  }
}
