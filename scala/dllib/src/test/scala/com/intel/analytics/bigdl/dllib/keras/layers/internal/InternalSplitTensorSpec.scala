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

import com.intel.analytics.bigdl.nn.JoinTable
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest
import org.scalatest.{FlatSpec, Matchers}

class InternalSplitTensorpec extends FlatSpec with Matchers {
  "A InternalSplitTensor" should "be able to work" in {
    val l1 = Tensor[Float](Array[Float](1, 2, 3, 4, 5, 6,
      1.2f, 2.2f, 3.2f, 4.2f, 5.2f, 6.2f), Array(2, 6))
    val l2 = Tensor[Float](Array[Float](1.1f, 2.1f, 3.1f, 4.1f, 5.1f, 6.1f,
      1.3f, 2.3f, 3.3f, 4.3f, 5.3f, 6.3f), Array(2, 6))
    val input = T(l1, l2)

    val layer = new JoinTable[Float](2, 2)
    val output = layer.forward(input).toTensor[Float]

    val layer2 = new InternalSplitTensor[Float](2, 2)
    val o2 = layer2.forward(output)

    val g1 = o2[Tensor[Float]](1)
    val g2 = o2[Tensor[Float]](2)
    assert(g1.almostEqual(l1, 1e-8) == true)
    assert(g2.almostEqual(l2, 1e-8) == true)

    val gradInput = layer2.backward(output, o2)
    assert(output.almostEqual(gradInput, 1e-8) == true)
  }
}

class InternalSplitTensorSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = new InternalSplitTensor[Float](2, 2).setName("InternalSplitTensor")
    val input = Tensor[Float](4, 4).rand()
    runSerializationTest(layer, input)
  }
}
