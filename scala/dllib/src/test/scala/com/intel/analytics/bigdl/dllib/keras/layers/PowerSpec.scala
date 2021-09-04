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

import com.intel.analytics.bigdl.nn.{Power => BPower}
import com.intel.analytics.zoo.pipeline.api.keras.layers.{Power => ZPower}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest


class PowerSpec extends ZooSpecHelper {

  "Power (2, 3) Zoo" should "be the same as BigDL" in {
    val blayer = BPower[Float](2)
    val zlayer = ZPower[Float](2, inputShape = Shape(3))
    zlayer.build(Shape(-1, 3))
    zlayer.getOutputShape().toSingle().toArray should be (Array(-1, 3))
    val input = Tensor[Float](Array(2, 3)).range(1, 6, 1)
    compareOutputAndGradInput(blayer, zlayer, input)
  }

  "Power (3, 3) Zoo" should "be the same as BigDL" in {
    val blayer = BPower[Float](3)
    val zlayer = ZPower[Float](3, inputShape = Shape(3))
    zlayer.build(Shape(-1, 3))
    zlayer.getOutputShape().toSingle().toArray should be (Array(-1, 3))
    val input = Tensor[Float](Array(3, 3)).range(1, 9, 1)
    compareOutputAndGradInput(blayer, zlayer, input)
  }

}

class PowerSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = ZPower[Float](2, inputShape = Shape(3))
    layer.build(Shape(2, 3))
    val input = Tensor[Float](2, 3).rand()
    runSerializationTest(layer, input)
  }
}
