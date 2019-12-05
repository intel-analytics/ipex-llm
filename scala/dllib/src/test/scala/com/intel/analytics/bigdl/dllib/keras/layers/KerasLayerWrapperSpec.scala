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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.{Linear, ReLU}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest


class KerasLayerWrapperSpec extends KerasBaseSpec {

  "kerasLayerWrapper" should "be test" in {
    val seq = Sequential[Float]()
    val dense = new KerasLayerWrapper[Float](Linear[Float](3, 2)
      .asInstanceOf[AbstractModule[Activity, Activity, Float]], inputShape = Shape(3))
    seq.add(dense)
    seq.add(Dense[Float](10))
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 10))
  }
}

class KerasLayerWrapperSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = new KerasLayerWrapper[Float](ReLU[Float]()
      .asInstanceOf[AbstractModule[Activity, Activity, Float]], inputShape = Shape(8, 12))
    layer.build(Shape(3, 8, 12))
    val input = Tensor[Float](3, 8, 12).rand()
    runSerializationTest(layer, input)
  }
}

