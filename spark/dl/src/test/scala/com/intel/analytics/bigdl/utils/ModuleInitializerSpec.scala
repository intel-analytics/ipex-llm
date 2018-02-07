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

package com.intel.analytics.bigdl.utils

import com.intel.analytics.bigdl.nn.Linear
import com.intel.analytics.bigdl.optim.L2Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{FlatSpec, Matchers}

class ModuleInitializerSpec extends FlatSpec with Matchers {

  "ModuleInitializer" should "init a Module with Default Constructor's Params correctly" in {
    val state = T("name" -> "Linear", "inputSize" -> 10, "outputSize" -> 5)
    var linear = ModuleInitializer.init[Float](state).asInstanceOf[Linear[Float]]
    linear.withBias shouldEqual true
    linear.forward(Tensor[Float](3, 10).rand()).size() shouldEqual Array(3, 5)

    state.update("withBias", false)
    state.update("wRegularizer", new L2Regularizer[Float](1e-3))
    state.update("initWeight", Tensor.ones[Float](5, 10))
    linear = ModuleInitializer.init[Float](state).asInstanceOf[Linear[Float]]
    linear.withBias shouldEqual false
    linear.wRegularizer should not be null
    linear.weight.storage().array().forall(_ == 1.0f) shouldEqual true
  }

}
