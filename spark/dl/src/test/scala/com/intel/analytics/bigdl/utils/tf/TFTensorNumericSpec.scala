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
package com.intel.analytics.bigdl.utils.tf

import com.google.protobuf.ByteString
import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{FlatSpec, Matchers}

class TFTensorNumericSpec extends FlatSpec with Matchers {

  import TFTensorNumeric.NumericByteString

  "String Tensor" should "works correctly" in {
    val a = Tensor[ByteString](Array(ByteString.copyFromUtf8("a"),
      ByteString.copyFromUtf8("b")), Array(2))
    val b = Tensor[ByteString](Array(ByteString.copyFromUtf8("a"),
      ByteString.copyFromUtf8("b")), Array(2))
    val sum = Tensor[ByteString](Array(ByteString.copyFromUtf8("aa"),
      ByteString.copyFromUtf8("bb")), Array(2))

    a + b should be (sum)
  }

}
