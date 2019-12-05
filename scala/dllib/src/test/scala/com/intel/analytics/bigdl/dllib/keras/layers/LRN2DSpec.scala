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

import com.intel.analytics.bigdl.nn.SpatialCrossMapLRN
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest


class LRN2DSpec extends ZooSpecHelper {

  "LRN2D Zoo th" should "be the same as BigDL" in {
    val blayer = SpatialCrossMapLRN[Float](5, 0.0001)
    val zlayer = LRN2D[Float](inputShape = Shape(3, 32, 32))
    zlayer.build(Shape(-1, 3, 32, 32))
    zlayer.getOutputShape().toSingle().toArray should be (Array(-1, 3, 32, 32))
    val input = Tensor[Float](Array(10, 3, 32, 32)).rand()
    compareOutputAndGradInput(blayer, zlayer, input)
  }

  "LRN2D Zoo tf" should "be the same as BigDL" in {
    val blayer = SpatialCrossMapLRN[Float](5, 0.001, 0.75, 2.0)
    val zlayer = LRN2D[Float](0.001, 2.0, 0.75, 5, inputShape = Shape(12, 12, 2))
    zlayer.build(Shape(-1, 12, 12, 2))
    zlayer.getOutputShape().toSingle().toArray should be (Array(-1, 12, 12, 2))
    val input = Tensor[Float](Array(10, 12, 12, 2)).rand()
    compareOutputAndGradInput(blayer, zlayer, input)
  }

}

class LRN2DSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = LRN2D[Float](inputShape = Shape(3, 32, 32))
    layer.build(Shape(2, 3, 32, 32))
    val input = Tensor[Float](2, 3, 32, 32).rand()
    runSerializationTest(layer, input)
  }
}
