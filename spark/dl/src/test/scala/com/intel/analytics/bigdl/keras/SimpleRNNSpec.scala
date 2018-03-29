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

package com.intel.analytics.bigdl.keras

import com.intel.analytics.bigdl.models.rnn.SimpleRNN
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator

class SimpleRNNSpec extends KerasBaseSpec {

  "SimpleRNN" should "generate the correct outputShape" in {
    val rnn = SimpleRNN.keras(4001, 40, 4001)
    rnn.getOutputShape().toSingle().toArray should be (Array(-1, 25, 4001))
  }

  "SimpleRNN Keras-Style definition" should "be the same as Torch-Style definition" in {
    RandomGenerator.RNG.setSeed(1000)
    val kmodel = SimpleRNN.keras(4001, 40, 4001)
    RandomGenerator.RNG.setSeed(1000)
    val tmodel = SimpleRNN(4001, 40, 4001)
    val input = Tensor[Float](Array(3, 25, 4001)).rand()
    compareModels(kmodel, tmodel, input)
  }

}
