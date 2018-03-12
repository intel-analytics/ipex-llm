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
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest

import scala.util.Random

class GRUSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    RNG.setSeed(100)
    val gru = GRU[Float](100, 100)
    val gruModel = Recurrent[Float]().add(gru).setName("gru")
    val input = Tensor[Float](2, 20, 100).apply1(e => Random.nextFloat())
    runSerializationTest(gruModel, input, gru.getClass)
  }
}
