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
package com.intel.analytics.bigdl.nn.tf

import com.google.protobuf.ByteString
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import com.intel.analytics.bigdl.utils.tf.TFTensorNumeric.NumericByteString

class AssertSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val assert = new Assert[Float]().setName("assert")
    val predictTensor = Tensor[Boolean](Array(1))
    predictTensor.setValue(1, true)
    val msg = Tensor[ByteString](Array(1))
    msg.setValue(1, ByteString.copyFromUtf8("must be true"))
    val input = T(predictTensor, msg)
    runSerializationTest(assert, input)
  }
}
