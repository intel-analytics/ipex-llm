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


class SpatialShareConvolutionSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val spatialShareConvolution = SpatialShareConvolution[Float](1, 1, 2, 2, 1, 1).
      setName("spatialShareConvolution")
    val input = Tensor[Float](3, 1, 3, 4).apply1( e => Random.nextFloat())
    runSerializationTest(spatialShareConvolution, input)
  }
}
