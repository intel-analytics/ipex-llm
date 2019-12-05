/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.pipeline.api.keras.layers.internal

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{RandomGenerator}
import com.intel.analytics.zoo.pipeline.api.keras.layers.KerasBaseSpec
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest


class InternalExpandSpec extends KerasBaseSpec {
  "InternalExpand" should "generate correct output" in {
    RandomGenerator.RNG.setSeed(100)
    val tgtSizes = Array(5, 4, 3)
    val layer = InternalExpand[Float](tgtSizes)
    val input = Tensor[Float](5, 4, 1).rand()
    val gradOutput = Tensor[Float](5, 4, 3).rand()
    val output = layer.forward(input)
    for (i <- 1 to 3) {
      require(output.narrow(3, i, 1).almostEqual(input, 1e-8) == true)
    }
    val gradInput = layer.backward(input, gradOutput)

    val expectGradInput = Tensor[Float](Array[Float](1.724836f, 1.2709723f, 0.33389157f,
      2.2362313f, 1.1102785f, 1.2065659f, 1.3381615f, 2.2813537f, 2.073785f, 1.0430324f,
      0.5571449f, 1.4316915f, 0.9529829f, 1.6594222f, 1.0953714f, 1.2271657f, 1.3691416f,
      1.7870495f, 1.5014464f, 1.4872407f), Array(5, 4, 1))
    require(gradInput.almostEqual(expectGradInput, 1e-8) == true)
  }

  "InternalExpand with -1 batch dim" should "generate correct output" in {
    RandomGenerator.RNG.setSeed(100)
    val tgtSizes = Array(-1, 4, 3)
    val layer = InternalExpand[Float](tgtSizes)
    val input = Tensor[Float](5, 4, 1).rand()
    val gradOutput = Tensor[Float](5, 4, 3).rand()
    val output = layer.forward(input)
    for (i <- 1 to 3) {
      require(output.narrow(3, i, 1).almostEqual(input, 1e-8) == true)
    }
    val gradInput = layer.backward(input, gradOutput)
    val expectGradInput = Tensor[Float](Array[Float](1.724836f, 1.2709723f, 0.33389157f,
      2.2362313f, 1.1102785f, 1.2065659f, 1.3381615f, 2.2813537f, 2.073785f, 1.0430324f,
      0.5571449f, 1.4316915f, 0.9529829f, 1.6594222f, 1.0953714f, 1.2271657f, 1.3691416f,
      1.7870495f, 1.5014464f, 1.4872407f), Array(5, 4, 1))
    require(gradInput.almostEqual(expectGradInput, 1e-8) == true)
  }
}

class InternalExpandSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val tgtSizes = Array(3, 2, 4)
    val layer = InternalExpand[Float](tgtSizes).setName("InternalExpand")
    val input = Tensor[Float](3, 2, 1).rand()
    runSerializationTest(layer, input)
  }
}
