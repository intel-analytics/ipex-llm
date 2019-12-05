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

import com.intel.analytics.bigdl.nn.{CMul => BCMul}
import com.intel.analytics.zoo.pipeline.api.keras.layers.{CMul => ZCMul}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest


class CMulSpec extends ZooSpecHelper {

  "CMul (2, 1) Zoo" should "be the same as BigDL" in {
    val blayer = BCMul[Float](Array(2, 1))
    val zlayer = ZCMul[Float](Array(2, 1), inputShape = Shape(3))
    zlayer.build(Shape(-1, 3))
    zlayer.getOutputShape().toSingle().toArray should be (Array(-1, 3))
    val input = Tensor[Float](Array(2, 3)).rand()
    compareOutputAndGradInputSetWeights(blayer, zlayer, input)
  }

  "CMul (1, 1, 1) Zoo" should "be the same as BigDL" in {
    val blayer = BCMul[Float](Array(1, 1, 1))
    val zlayer = ZCMul[Float](Array(1, 1, 1), inputShape = Shape(3, 4))
    zlayer.build(Shape(-1, 3, 4))
    zlayer.getOutputShape().toSingle().toArray should be (Array(-1, 3, 4))
    val input = Tensor[Float](Array(2, 3, 4)).rand()
    compareOutputAndGradInputSetWeights(blayer, zlayer, input)
  }

}

class CMulSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = ZCMul[Float](Array(1, 1, 1), inputShape = Shape(3, 4))
    layer.build(Shape(2, 3, 4))
    val input = Tensor[Float](2, 3, 4).rand()
    runSerializationTest(layer, input)
  }
}
