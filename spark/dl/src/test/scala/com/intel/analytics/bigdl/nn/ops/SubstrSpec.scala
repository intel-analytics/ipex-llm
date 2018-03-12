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
package com.intel.analytics.bigdl.nn.ops

import com.google.protobuf.ByteString
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.scalatest.{FlatSpec, Matchers}

class SubstrSpec extends FlatSpec with Matchers {
  "Substr operation" should "works correctly" in {
    import com.intel.analytics.bigdl.utils.tf.TFTensorNumeric.NumericByteString
    val data = Tensor.scalar(ByteString.copyFromUtf8("abc"))
    val pos = Tensor.scalar(0)
    val len = Tensor.scalar(2)
    val expectOutput = Tensor.scalar(ByteString.copyFromUtf8("ab"))

    val output = Substr().forward(T(data, pos, len))
    output should be(expectOutput)
  }
}

class SubstrSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    import com.intel.analytics.bigdl.utils.tf.TFTensorNumeric.NumericByteString
    val subStr = Substr[Float]().setName("subStr")
    val input = T(Tensor.scalar[ByteString](ByteString.copyFromUtf8("HelloBigDL")),
      Tensor.scalar[Int](0), Tensor.scalar[Int](5))
    runSerializationTest(subStr, input)
  }
}
