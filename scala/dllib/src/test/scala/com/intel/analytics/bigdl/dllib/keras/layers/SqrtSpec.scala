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

package com.intel.analytics.zoo.pipeline.api.keras.layers

import com.intel.analytics.bigdl.nn.{Sqrt => BSqrt}
import com.intel.analytics.zoo.pipeline.api.keras.layers.{Sqrt => ZSqrt}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest


class SqrtSpec extends ZooSpecHelper {

  "Sqrt input size (3, 5) Zoo" should "be the same as BigDL" in {
    val blayer = BSqrt[Float]()
    val zlayer = ZSqrt[Float](inputShape = Shape(5))
    zlayer.build(Shape(-1, 5))
    zlayer.getOutputShape().toSingle().toArray should be (Array(-1, 5))
    val input = Tensor[Float](Array(3, 5)).range(1, 15, 1)
    compareOutputAndGradInput(blayer, zlayer, input)
  }

  "Sqrt input size (3, 3) Zoo" should "be the same as BigDL" in {
    val blayer = BSqrt[Float]()
    val zlayer = ZSqrt[Float](inputShape = Shape(3))
    zlayer.build(Shape(-1, 3))
    zlayer.getOutputShape().toSingle().toArray should be (Array(-1, 3))
    val input = Tensor[Float](Array(3, 3)).range(1, 9, 1)
    compareOutputAndGradInputSetWeights(blayer, zlayer, input)
  }

}

class SqrtSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = ZSqrt[Float](inputShape = Shape(3))
    layer.build(Shape(2, 3))
    val input = Tensor[Float](2, 3).rand()
    runSerializationTest(layer, input)
  }
}
