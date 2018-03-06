/*
 * Copyright 2016 The BigDL Authors.
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
package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest

import scala.util.Random


class VolumetricAveragePoolingSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val volumetricAveragePooling = VolumetricAveragePooling[Float](2, 2, 2, 1, 1, 1, 0, 0, 0).
      setName("volumetricAveragePooling")
    val input = Tensor[Float](1, 2, 3, 3).apply1(_ => Random.nextFloat())
    runSerializationTest(volumetricAveragePooling, input)
  }
}
