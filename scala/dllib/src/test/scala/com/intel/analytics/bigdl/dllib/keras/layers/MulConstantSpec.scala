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

import com.intel.analytics.bigdl.nn.{MulConstant => BMulConstant}
import com.intel.analytics.zoo.pipeline.api.keras.layers.{MulConstant => ZMulConstant}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest


class MulConstantSpec extends ZooSpecHelper {

  "MulConstant 0 Zoo" should "be the same as BigDL" in {
    val blayer = BMulConstant[Float](0f)
    val zlayer = ZMulConstant[Float](0f, inputShape = Shape(4, 5))
    zlayer.build(Shape(-1, 4, 5))
    zlayer.getOutputShape().toSingle().toArray should be (Array(-1, 4, 5))
    val input = Tensor[Float](Array(3, 4, 5)).rand()
    compareOutputAndGradInput(blayer, zlayer, input)
  }

  "MulConstant -1 Zoo" should "be the same as BigDL" in {
    val blayer = BMulConstant[Float](-1)
    val zlayer = ZMulConstant[Float](-1, inputShape = Shape(4, 8, 8))
    zlayer.build(Shape(-1, 4, 8, 8))
    zlayer.getOutputShape().toSingle().toArray should be (Array(-1, 4, 8, 8))
    val input = Tensor[Float](Array(3, 4, 8, 8)).rand()
    compareOutputAndGradInput(blayer, zlayer, input)
  }

}

class MulConstantSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = ZMulConstant[Float](-1, inputShape = Shape(4, 8, 8))
    layer.build(Shape(2, 4, 8, 8))
    val input = Tensor[Float](2, 4, 8, 8).rand()
    runSerializationTest(layer, input)
  }
}
