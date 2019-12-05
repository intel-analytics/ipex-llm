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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{MultiShape, Shape, T}
import com.intel.analytics.zoo.pipeline.api.keras.models._
import com.intel.analytics.zoo.pipeline.api.keras.layers.Merge.merge
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest


class MaxSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = Max[Float](1, inputShape = Shape(4, 8, 8))
    layer.build(Shape(2, 4, 8, 8))
    val input = Tensor[Float](2, 4, 8, 8).rand()
    runSerializationTest(layer, input)
  }
}
