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

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.nn.{Negative => BNegative}
import com.intel.analytics.zoo.pipeline.api.keras.layers.{Negative => ZNegative}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest


class NegativeSpec extends ZooSpecHelper {

  "Negative 3D Zoo" should "be the same as BigDL" in {
    val blayer = BNegative[Float]()
    val zlayer = ZNegative[Float](inputShape = Shape(3, 4))
    zlayer.build(Shape(-1, 3, 4))
    zlayer.getOutputShape().toSingle().toArray should be (Array(-1, 3, 4))
    val input = Tensor[Float](Array(2, 3, 4)).rand()
    compareOutputAndGradInput(
      blayer.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]], zlayer, input)
  }

  "Negative 4D Zoo" should "be the same as BigDL" in {
    val blayer = BNegative[Float]()
    val zlayer = ZNegative[Float](inputShape = Shape(4, 8, 8))
    zlayer.build(Shape(-1, 4, 8, 8))
    zlayer.getOutputShape().toSingle().toArray should be (Array(-1, 4, 8, 8))
    val input = Tensor[Float](Array(3, 4, 8, 8)).rand()
    compareOutputAndGradInput(
      blayer.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]], zlayer, input)
  }

}

class NegativeSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = Negative[Float](inputShape = Shape(4, 8))
    layer.build(Shape(2, 4, 8))
    val input = Tensor[Float](2, 4, 8).rand()
    runSerializationTest(layer, input)
  }
}
