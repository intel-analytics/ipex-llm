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
import com.intel.analytics.bigdl.utils.{MultiShape, Shape, SingleShape, T}
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest

class SelectTableSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val shape1 = SingleShape(List(2))
    val shape2 = SingleShape(List(2))

    val shape3 = SingleShape(List(1, 2))
    val shape4 = SingleShape(List(1, 2))

    val mul1 = MultiShape(List(shape1, shape2))
    val mul2 = MultiShape(List(shape3, shape4))
    val layer = SelectTable[Float](0, mul1)
    layer.build(mul2)
    val input = T(Tensor[Float](1, 2).rand(), Tensor[Float](1, 2).rand())
    runSerializationTest(layer, input)
  }
}
