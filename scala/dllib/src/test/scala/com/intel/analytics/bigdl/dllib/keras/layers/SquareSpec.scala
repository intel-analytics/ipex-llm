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

import com.intel.analytics.bigdl.nn.{Square => BSquare}
import com.intel.analytics.zoo.pipeline.api.keras.layers.{Square => ZSquare}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest


class SquareSpec extends ZooSpecHelper {

  "Square input size (1, 3) Zoo" should "be the same as BigDL" in {
    val blayer = BSquare[Float]()
    val zlayer = ZSquare[Float](inputShape = Shape(3))
    zlayer.build(Shape(-1, 3))
    zlayer.getOutputShape().toSingle().toArray should be (Array(-1, 3))
    val input = Tensor[Float](Array(1, 3)).range(1, 3, 1)
    compareOutputAndGradInput(blayer, zlayer, input)
  }

  "Square input size (2, 5) Zoo" should "be the same as BigDL" in {
    val blayer = BSquare[Float]()
    val zlayer = ZSquare[Float](inputShape = Shape(5))
    zlayer.build(Shape(-1, 5))
    zlayer.getOutputShape().toSingle().toArray should be (Array(-1, 5))
    val input = Tensor[Float](Array(2, 5)).range(2, 12, 1)
    compareOutputAndGradInput(blayer, zlayer, input)
  }

}

class SquareSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = Square[Float](inputShape = Shape(5))
    layer.build(Shape(2, 5))
    val input = Tensor[Float](2, 5).rand()
    runSerializationTest(layer, input)
  }
}
