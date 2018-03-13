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

class ConvLSTMPeepholeSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val hiddenSize = 5
    val inputSize = 3
    val seqLength = 4
    val batchSize = 2
    val kernalW = 3
    val kernalH = 3
    val c2d = ConvLSTMPeephole[Float](
      inputSize,
      hiddenSize,
      kernalW, kernalH,
      1,
      withPeephole = false)
    val convLSTMPeephole2d = Recurrent[Float]().setName("convLSTMPeephole2d")
    val model = Sequential[Float]()
      .add(convLSTMPeephole2d
        .add(c2d))
      .add(View[Float](hiddenSize * kernalH * kernalW))

    val input = Tensor[Float](batchSize, seqLength, inputSize, kernalW, kernalH).rand
    runSerializationTest(convLSTMPeephole2d, input, c2d.getClass)
  }
}
