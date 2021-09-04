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

import com.intel.analytics.bigdl.nn.Sequential
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.pipeline.api.keras.layers.internal.InternalRecurrent
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest

class InternalConvLSTM3DSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val hiddenSize = 5
    val inputSize = 3
    val seqLength = 4
    val batchSize = 2
    val kernal = 2
    val rec = new InternalRecurrent[Float]()
    val model = Sequential[Float]()
      .add(rec
        .add(new InternalConvLSTM3D[Float](
          inputSize,
          hiddenSize,
          kernal,
          withPeephole = true)))

    val input = Tensor[Float](batchSize, seqLength, inputSize, 5, 5, 5).rand
    runSerializationTest(model, input)
  }
}
